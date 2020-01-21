import cv2
import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO
from openvino.inference_engine import IENetwork, IEPlugin
from timeit import default_timer as timer
import numpy as np

logger = getLogger(__name__)
basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

is_myriad_plugin_initialized = False
myriad_plugin = None
is_cpu_plugin_initialized = False
cpu_plugin = None

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


class BaseDetection(object):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 detection_of):

        # Each device's plugin should be initialized only once,
        # MYRIAD plugin would be failed when createting exec_net(plugin.load method)
        # Error: "RuntimeError: Can not init USB device: NC_DEVICE_NOT_FOUND"
        global is_myriad_plugin_initialized
        global myriad_plugin
        global is_cpu_plugin_initialized
        global cpu_plugin

        if device == 'MYRIAD' and not is_myriad_plugin_initialized:
            self.plugin = self._init_plugin(device, cpu_extension, plugin_dir)
            is_myriad_plugin_initialized = True
            myriad_plugin = self.plugin
        elif device == 'MYRIAD' and is_myriad_plugin_initialized:
            self.plugin = myriad_plugin
        elif device == 'CPU' and not is_cpu_plugin_initialized:
            self.plugin = self._init_plugin(device, cpu_extension, plugin_dir)
            is_cpu_plugin_initialized = True
            cpu_plugin = self.plugin
        elif device == 'CPU' and is_cpu_plugin_initialized:
            self.plugin = cpu_plugin
        else:
            self.plugin = self._init_plugin(device, cpu_extension, plugin_dir)

        # Read IR
        self.net = self._read_ir(model_xml, detection_of)
        # Load IR model to the plugin
        self.input_blob, self.out_blob, self.exec_net, self.input_dims, self.output_dims = self._load_ir_to_plugin(
            device, detection_of)

    def _init_plugin(self, device, cpu_extension, plugin_dir):
        logger.info("Initializing plugin for {} device...".format(device))
        plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
        logger.info(
            "Plugin for {} device version:{}".format(device, plugin.version))
        if cpu_extension and 'CPU' in device:
            plugin.add_cpu_extension(cpu_extension)
        return plugin

    def _read_ir(self, model_xml, detection_of):
        logger.info("Reading IR Loading for {}...".format(detection_of))
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        return IENetwork(model=model_xml, weights=model_bin)

    def _load_ir_to_plugin(self, device, detection_of):
        if device == "CPU" and detection_of == "Face Detection":
            supported_layers = self.plugin.get_supported_layers(self.net)
            not_supported_layers = [
                l for l in self.net.layers.keys() if l not in supported_layers
            ]
            if len(not_supported_layers) != 0:
                logger.error(
                    "Following layers are not supported by the plugin for specified device {}:\n {}".
                    format(self.plugin.device, ', '.join(
                        not_supported_layers)))
                logger.error(
                    "Please try to specify cpu extensions library path in demo's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)
        if detection_of == "Face Detection":
            logger.info("Checking Face Detection network inputs")
            assert len(self.net.inputs.keys(
            )) == 1, "Face Detection network should have only one input"
            logger.info("Checking Face Detection network outputs")
            assert len(
                self.net.outputs
            ) == 1, "Face Detection network should have only one output"

        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        logger.info("Loading {} model to the {} plugin...".format(
            device, detection_of))
        exec_net = self.plugin.load(network=self.net, num_requests=2)
        input_dims = self.net.inputs[input_blob].shape
        output_dims = self.net.outputs[out_blob].shape
        logger.info("{} input dims:{} output dims:{} ".format(
            detection_of, input_dims, output_dims))
        return input_blob, out_blob, exec_net, input_dims, output_dims

    def submit_req(self, face_frame, next_face_frame, is_async_mode):
        n, c, h, w = self.input_dims

        if is_async_mode:
            logger.debug(
                "*** start_async *** cur_req_id:{} next_req_id:{} async:{}".
                format(self.cur_request_id, self.next_request_id,
                       is_async_mode))
            in_frame = cv2.resize(next_face_frame, (w, h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((n, c, h, w))
            self.exec_net.start_async(
                request_id=self.next_request_id,
                inputs={self.input_blob: in_frame})
        else:
            logger.debug(
                "*** start_sync *** cur_req_id:{} next_req_id:{} async:{}".
                format(self.cur_request_id, self.next_request_id,
                       is_async_mode))
            self.exec_net.requests[self.cur_request_id].wait(-1)
            in_frame = cv2.resize(face_frame, (w, h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((n, c, h, w))
            self.exec_net.start_async(
                request_id=self.cur_request_id,
                inputs={self.input_blob: in_frame})

    def wait(self):
        logger.debug("*** start wait:{} ***".format(self.exec_net.requests[
            self.cur_request_id].wait(-1)))

        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            return True
        else:
            return False


class FaceDetection(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 prob_threshold_face, is_async_mode):
        self.prob_threshold_face = prob_threshold_face
        detection_of = "Face Detection"
        super().__init__(device, model_xml, cpu_extension, plugin_dir,
                         detection_of)

        del self.net
        self.cur_request_id = 0
        self.next_request_id = 1

    def get_results(self, is_async_mode):
        """
        The net outputs a blob with shape: [1, 1, 200, 7]
        The description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        """

        faces = None

        res = self.exec_net.requests[self.cur_request_id].outputs[
            self.out_blob]  # res's shape: [1, 1, 200, 7]

        # Get rows whose confidence is larger than prob_threshold.
        # detected faces are also used by age/gender, emotion, landmark, head pose detection.
        faces = res[0][:, np.where(res[0][0][:, 2] > self.prob_threshold_face)]

        if is_async_mode:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

        return faces


class AgeGenderDetection(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 prob_threshold, is_async_mode):
        detection_of = "Age/Gender Detection"
        super().__init__(device, model_xml, cpu_extension, plugin_dir,
                         detection_of)
        self.label = ('F', 'M')
        del self.net
        self.cur_request_id = 0
        self.next_request_id = 1

    def get_results(self, is_async_mode):
        """
        Output layer names in Inference Engine format:
         "age_conv3", shape: [1, 1, 1, 1] - Estimated age divided by 100.
         "prob", shape: [1, 2, 1, 1] - Softmax output across 2 type classes [female, male]
        """
        age = 0
        gender = ""
        logger.debug("*** get_results start *** cur_request_id:{}".format(
            self.cur_request_id))
        age = self.exec_net.requests[self.cur_request_id].outputs['age_conv3']
        prob = self.exec_net.requests[self.cur_request_id].outputs['prob']
        age = age[0][0][0][0] * 100
        gender = self.label[np.argmax(prob[0])]
        if is_async_mode:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id
        return age, gender


class SSDetection(BaseDetection):
    def __init__(self, device, model_xml, cpu_extension, plugin_dir,
                 prob_threshold, is_async_mode):
        detection_of = "MobileNet-SSD Detection"
        super().__init__(device, model_xml, cpu_extension, plugin_dir,
                         detection_of)
        del self.net
        self.prob_threshold = prob_threshold
        self.cur_request_id = 0
        self.next_request_id = 1

    def object_inference(self, frame, next_frame, is_async_mode):
        n, c, h, w = self.input_dims
        frame_h, frame_w = frame.shape[:2]  # shape (h, w, c)
        det_time = 0

        # Main sync point:
        # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately wait for it's completion
        inf_start = timer()
        if is_async_mode:
            in_frame = cv2.dnn.blobFromImage(
                cv2.resize(next_frame, (300, 300)), 0.007843, (300, 300),
                127.5)
            self.exec_net.start_async(
                request_id=self.next_request_id,
                inputs={self.input_blob: in_frame})
        else:
            if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
                in_frame = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                self.exec_net.start_async(
                    request_id=self.cur_request_id,
                    inputs={self.input_blob: in_frame})
        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            inf_end = timer()
            det_time = inf_end - inf_start
            # Parse detection results of the current request
            logger.debug("computing object detections...")
            det_objects = self.exec_net.requests[self.cur_request_id].outputs[
                self.out_blob]
            # loop over the detections
            for i in np.arange(0, det_objects.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = det_objects[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence > self.prob_threshold:
                    # extract the index of the class label from the `detections`, then compute the (x, y)-coordinates
                    # of the bounding box for the object
                    idx = int(det_objects[0, 0, i, 1])
                    box = det_objects[0, 0, i, 3:7] * np.array(
                        [frame_w, frame_h, frame_w, frame_h])
                    (startX, startY, endX, endY) = box.astype("int")
                    logger.debug("startX, startY, endX, endY: {}".format(
                        box.astype("int")))

                    # display the prediction
                    label = "{}: {:.2f}%".format(CLASSES[idx],
                                                 confidence * 100)
                    logger.debug("{} {}".format(self.cur_request_id, label))
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        if is_async_mode:
            self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

        return det_time, frame
