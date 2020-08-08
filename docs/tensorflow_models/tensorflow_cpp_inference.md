## Why and How do we have Tensorflow Inference Engine

Typically there are plentfy of ways to run deep learning models and blend them into industrial softwares.   

But I found only few of them suitable for our application:

1. Export python codes to c++: Most of existing models are developed in python and our main program is developped in c++ for real time application (slam).  
However Python runtime is very expensive to call from c++, especially inside a forever true query loop.
2. Passing messages through communication (network, pipe or filesystem):
     2.1 Deploy subscribers and publishers network with cross language message transportation (grpc:tcp+http2.0): Another method is to export services and host programs can fetch the results through network infra.  
This is a big project I used to have done with grpc:  
1) a node (python) subscribes image sources and publishes results to a broker developed using c++  
2) and our application (slam) publishes image sources and subscribes results of the predictions from the python node  
     2.2. Depoly a prediction server(http/https 1.1): This is easy to implement but migth not work with a complex model, for example, MRCNN, that we have to compute anchors, preprocess images, fetch and filter results from the predictions.
3. C++ inference engine: Export the model and variables to neutral format to describe the graph and provide customer codes(c++) to preprocess images, fetch and filter results from the predictions. The 


Both quality of software implementation on network infra and sizes of messages to pass, greatly affect the performance of inferences. With concern of performance of inferences in real time, and complexity involving software implementation of network infra, we provides c++ inference engine for cpp implementation for the moment. 

## Tensorflow/Keras model exportation and saved format 

Tensorflow has upgraded to tf 2.2.0(2020.6, upgraded from 2.2.0-rc 2020.3) Though many codes are developed base on tensorflow 1.x. syntax and codebase, tf 2.0 is better supported by the latest cuda and nvidia gpu drivers.

To make all things work together, we use tensorflow 2.2.0 and disable v2 behaviors:

```python
import tensorflow as tf
print("tensorflow ver: ", tf.__version__)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

Model using in POC stage is MRCNN with weights trained from coco outdoor dataset. I also tested it in indoor dataset before use and the model works very well. Since the original
MRCNN was developed in python, most of exportation is done with python\(virtual env: py36\).

#### Export Keras computation grpah to protobuffer file

There are plenty of methods explored in "pysvso/models/sfe.py" to export keras models computation graph to tensorflow protobuffer file. Earlier days, one can convert keras variables to constant with "tf.graph_util.convert_variables_to_constants", append them to tensorflow graph and either with "tensorflow.train.write_graph" or "tensorflow.io.gfile.GFile" to dump the created tensorflow graph to a protobuffer file.

```python
# pysvso.sfe.freeze_session
# @see https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

# pysvso.sfe.Program
    K.set_learning_phase(0)
    target_node_names = [out.op.name for out in sfe.get_base_model().keras_model.outputs]
    [print(name) for name in target_node_names]
    frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=target_node_names)
    tf.train.write_graph(frozen_graph, os.path.join(sfe_config.MODEL_PATH, "coco/mrcnn_tmp"), "mrcnn_tmp.pb", as_text=False)
```

The following codes is equivalent
```python
# @todo TODO (status: not safe to use)
def save_model(model, export_path, no_custom_op=False):
    if os.path.isdir(export_path):
        logging.info("%s already exits, deleting it ...")
        os.system("rm -rf %s" % export_path)
        os.mkdir(export_path)
    else:
        os.mkdir(export_path)

    # ?
    K.set_learning_phase(0)

    tf1_to_2 = os.path.join(export_path, "tmp.h5")
    model.keras_model.save(tf1_to_2)

    if no_custom_op:
        # since we disable tf2 behaviors, we pass the flag manually. Caution! DO NOT USE "model.keras_model.save", this call
        # leads to a different implementation from that of "tensorflow/python/keras/saving/save.py". see SavedModel file
        # architectures.
        # loaded = tf.keras.models.load_model(tf1_to_2)

        # @todo : TODO(does not work)  exception with "Unknown layer: BatchNorm" MRCNN has custom layers and is not supported by TF or TFLite, sadly
        # tf.keras.models.save_model(loaded, export_path, save_format='tf')

        # hence we have to write the graph to files manually with the add of "tensorflow.gfile.GFile" toolkit.
        # once you build tensorflow from source (see docs/install/build_tensorflow_from_source.md), you will have binary named 'save_model_cli'
        #   save_model_cli show --dir log/models/coco/mrcnn --all
        # gives you the runtime name of target operators.

        # deprecated, the names not valid for tensorflow
        target_node_names = [op.name for op in model.keras_model.outputs]
    else:

        target_node_names = ["detections", "mrcnn_class", "mrcnn_bbox", "mrcnn_mask", "rois", "rpn_class", "rpn_bbox"]

    target_node_names = [ "output_" + name for name in target_node_names ]
    [tf.identity(model.keras_model.outputs[i], name = target_node_names[i]) for i in range(len(target_node_names))] # add to tensorflow graph

    if is_tf_1():
        Session = K.get_session
        gfile = tf.gfile
    else:
        Session = tf.keras.backend.get_session # tf.Session
        gfile = tf.io.gfile

    with Session() as sess:
        # freeze the graph and convert variables to constants
        graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), target_node_names)

        with gfile.GFile(os.path.join(export_path, "mrcnn.pb"), 'wb') as f:
            f.write(graph_def.SerializeToString())

    logging.info("saving model to %s ..." % export_path)

```

show the export directory, we have following files structure:

```
(py36) ➜  coco git:(dev/wangyi/feature_add_cpp_impl) ✗ ls
mrcnn  mrcnn_tmp  sfe
(py36) ➜  coco git:(dev/wangyi/feature_add_cpp_impl) ✗ ls mrcnn_tmp 
mrcnn.pb  mrcnn_tmp.pb  tmp.h5

```

#### Export Keras model to new model format for serving

The question could be traced back to 2017 in google cloud forum. The engineer provides a solution (not mature at that moment) using serving api "saved_model.Builder". With tensorflow 2.0, serving api is now native to tensorflow.keras to dump and restore models either in "h5" or "tf" format. With native keras models not built from "tensorflow.keras" we have additional works to do:

```python
def save_mrcnn_to_serving(model, export_path):
    if os.path.isdir(export_path):
        logging.info("%s already exits, deleting it ...")
        os.system("rm -rf %s" % export_path)

    # Tensorflow 2.2.0-rc
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        # bboxes, molded_images, image_metas, anchors
        inputs={
            'input_image': model.inputs[0],
            'input_image_meta': model.inputs[1],
            'input_anchors': model.inputs[2]},
        outputs={
            'detections': model.outputs[0], # important!
            'class': model.outputs[1],
            'bbox': model.outputs[2],
            'mask': model.outputs[3], # important!
            'rpn_rois': model.outputs[4],
            'rpn_class': model.outputs[5],
            'rpn_bbox': model.outputs[6]
        }
    )

    legacy_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    if is_tf_1():
        Session = K.get_session
    else:
        Session = tf.keras.backend.get_session # tf.Session

    # with Session(graph=tf.Graph()) as sess:
    with Session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            },
            legacy_init_op=legacy_op
        )
        logging.info("saving mrcnn model to %s ..." % export_path)
        builder.save()
```

show the export directory, we have following files structure:

```
(py36) ➜  coco git:(dev/wangyi/feature_add_cpp_impl) ✗ ls
mrcnn  mrcnn_tmp  sfe
(py36) ➜  coco git:(dev/wangyi/feature_add_cpp_impl) ✗ ls mrcnn
saved_model.pb  variables
(py36) ➜  coco git:(dev/wangyi/feature_add_cpp_impl) ✗ ls mrcnn/variables 
variables.data-00000-of-00001  variables.index

```

## Tensorflow cpp inference engine

At the bottom of engine, we construct tensorflow session with imported graph definitions. At the top of engine, we implemented various models which handles methods to load graph definitions and populate data into and extract predictions from containers of type "tf::Tensor".

Since we have two formats of stored models, two apis are provided in engine to load models. Here is one of the running examples of the engine:

```
/home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/cmake-build-debug/bin/simple_mrcnn_infer
2020-07-22 22:23:01.564987: I /home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/modules/models/engine.h:208] [TensorFlowEngine.load_graph] loading </home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/log/models/coco/mrcnn> ...
2020-07-22 22:23:01.565027: I tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/log/models/coco/mrcnn
2020-07-22 22:23:01.580922: I tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2020-07-22 22:23:01.580957: I tensorflow/cc/saved_model/loader.cc:295] Reading SavedModel debug info (if present) from: /home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/log/models/coco/mrcnn
2020-07-22 22:23:01.581068: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-22 22:23:01.593693: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-07-22 22:23:01.783570: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:01.783931: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce RTX 2080 Ti computeCapability: 7.5
coreClock: 1.65GHz coreCount: 68 deviceMemorySize: 10.76GiB deviceMemoryBandwidth: 573.69GiB/s
2020-07-22 22:23:01.783957: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:01.784275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 1 with properties: 
pciBusID: 0000:05:00.0 name: GeForce RTX 2080 Ti computeCapability: 7.5
coreClock: 1.65GHz coreCount: 68 deviceMemorySize: 10.76GiB deviceMemoryBandwidth: 573.69GiB/s
2020-07-22 22:23:01.789275: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.2
2020-07-22 22:23:01.796721: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-07-22 22:23:01.802124: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2020-07-22 22:23:01.806566: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2020-07-22 22:23:01.814083: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-07-22 22:23:01.819951: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2020-07-22 22:23:01.828092: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-07-22 22:23:01.828166: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:01.828604: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:01.828959: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:01.829303: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:01.829637: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0, 1
2020-07-22 22:23:02.389703: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-22 22:23:02.389729: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0 1 
2020-07-22 22:23:02.389734: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N N 
2020-07-22 22:23:02.389737: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 1:   N N 
2020-07-22 22:23:02.389888: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:02.390257: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:02.390653: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:02.391036: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:02.391420: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9697 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5)
2020-07-22 22:23:02.391997: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:02.392491: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:02.392870: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10076 MB memory) -> physical GPU (device: 1, name: GeForce RTX 2080 Ti, pci bus id: 0000:05:00.0, compute capability: 7.5)
2020-07-22 22:23:02.420670: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 3000000000 Hz
2020-07-22 22:23:02.422066: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b3a8bb0310 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-22 22:23:02.422079: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-07-22 22:23:02.423214: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b3a8b0ef10 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-07-22 22:23:02.423226: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce RTX 2080 Ti, Compute Capability 7.5
2020-07-22 22:23:02.423230: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (1): GeForce RTX 2080 Ti, Compute Capability 7.5
2020-07-22 22:23:02.557067: I tensorflow/cc/saved_model/loader.cc:234] Restoring SavedModel bundle.
2020-07-22 22:23:03.145362: I tensorflow/cc/saved_model/loader.cc:364] SavedModel load for tags { serve }; Status: success: OK. Took 1580333 microseconds.
2020-07-22 22:23:03.146042: I /home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/modules/models/engine.h:208] [TensorFlowEngine.load_graph] loading </home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/log/models/coco/mrcnn> ...
2020-07-22 22:23:03.146060: I tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/log/models/coco/mrcnn
2020-07-22 22:23:03.158047: I tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2020-07-22 22:23:03.158061: I tensorflow/cc/saved_model/loader.cc:295] Reading SavedModel debug info (if present) from: /home/yiak/WorkSpace/Github/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY/log/models/coco/mrcnn
2020-07-22 22:23:03.158125: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:03.158352: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce RTX 2080 Ti computeCapability: 7.5
coreClock: 1.65GHz coreCount: 68 deviceMemorySize: 10.76GiB deviceMemoryBandwidth: 573.69GiB/s
2020-07-22 22:23:03.158378: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:03.158693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 1 with properties: 
pciBusID: 0000:05:00.0 name: GeForce RTX 2080 Ti computeCapability: 7.5
coreClock: 1.65GHz coreCount: 68 deviceMemorySize: 10.76GiB deviceMemoryBandwidth: 573.69GiB/s
2020-07-22 22:23:03.158710: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.2
2020-07-22 22:23:03.158715: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-07-22 22:23:03.158720: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2020-07-22 22:23:03.158725: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2020-07-22 22:23:03.158731: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-07-22 22:23:03.158736: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2020-07-22 22:23:03.158741: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-07-22 22:23:03.158759: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:03.158964: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:03.159285: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:03.159489: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:03.159796: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0, 1
2020-07-22 22:23:03.159841: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-22 22:23:03.159848: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0 1 
2020-07-22 22:23:03.159851: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N N 
2020-07-22 22:23:03.159853: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 1:   N N 
2020-07-22 22:23:03.159903: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:03.160111: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:03.160479: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:03.160678: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9697 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5)
2020-07-22 22:23:03.160703: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-07-22 22:23:03.161014: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10076 MB memory) -> physical GPU (device: 1, name: GeForce RTX 2080 Ti, pci bus id: 0000:05:00.0, compute capability: 7.5)
2020-07-22 22:23:03.278299: I tensorflow/cc/saved_model/loader.cc:234] Restoring SavedModel bundle.
2020-07-22 22:23:03.761375: I tensorflow/cc/saved_model/loader.cc:364] SavedModel load for tags { serve }; Status: success: OK. Took 615315 microseconds.
```

The most difficul things about this part is that we have documentation in tensorflow official websitea about training and inference using c++ api, and you have to dig into source codes to see usage examples, definitions and make a way how to use them.



#### Load Exported Computation Graph

Here is an example:

```c++
class TF_MRCNN_SemanticFeatureExtractor : public Model {
public:
    using Ptr = std::shared_ptr<TF_MRCNN_SemanticFeatureExtractor>;
    using ConstPtr = std::shared_ptr<const TF_MRCNN_SemanticFeatureExtractor>;

    TF_MRCNN_SemanticFeatureExtractor() : config_(InferenceConfig()) {
        base_engine_.reset(new TensorFlowEngine() );
        const std::string base_model_dir = format(
                "%s/%s",
                config_.MODEL_DIR.c_str(),
                config_.BASE_MODEL_DIR.c_str()
        );
        base_model_dir_ = base_model_dir;
	...
        const std::string base_model_graph = format(
                "%s/%s",
                base_model_dir.c_str(),
                config_.BASE_GRAPH_DEF.c_str()
        );
        base_engine_->load_graph(base_model_graph);
        base_engine_->Init();
	...

    }

```

#### Load Exported Models

Here is an another example:

```c++
class TF_MRCNN_SemanticFeatureExtractor : public Model {
public:
    using Ptr = std::shared_ptr<TF_MRCNN_SemanticFeatureExtractor>;
    using ConstPtr = std::shared_ptr<const TF_MRCNN_SemanticFeatureExtractor>;

    TF_MRCNN_SemanticFeatureExtractor() : config_(InferenceConfig()) {
        base_engine_.reset(new TensorFlowEngine() );
        const std::string base_model_dir = format(
                "%s/%s",
                config_.MODEL_DIR.c_str(),
                config_.BASE_MODEL_DIR.c_str()
        );
        base_model_dir_ = base_model_dir;
        base_engine_->load_saved_model(base_model_dir);
	...

    }

```

## Run inference

Once you loaded exported model using either of the methods, the program will print graph in console and create a session object. Only things you need to do are:

1. filling in tensors
2. extract and filter results from predictions


```cc
// modules/models/sfe.hpp
       if (inputs == nullptr || outputs == nullptr) {
            LOG(FATAL) << "[Main] Wrong Value: inputs and outpus should not be null" << std::endl;
        }
        inputs->clear();
        outputs->clear();

        // In c++ it is also possible for tensorflow to create an reader operator to automatically read images from an image
        // path, where image tensor is built automatically and graph_def is finally converted from a variable of type tf::Scope.
        // In tensorflow, see codes defined in "tensorflow/core/framework/tensor_types.h" and "tensorflow/core/framework/tensor.h"
        // that users are able to use Eigen::TensorMap to extract values from the container for reading and assignment. (Lei (yiak.wy@gmail.com) 2020.7)
        tfe::Tensor _molded_images(tf::DT_FLOAT, tf::TensorShape({1, molded_shape(0), molded_shape(1), 3}));
        auto _molded_images_mapped = _molded_images.tensor<float, 4>();
        // @todo TODO using Eigen::TensorMap to optimize the copy operation, e.g.: float* data_mapped = _molded_images.flat<float>().data();  copy to the buf using memcpy
        //   ref: 1. discussion Tensorflow Github repo issue#8033
        //        2. opencv2 :
        //          2.1. grab buf: Type* buf = mat.ptr<Type>();
        //          2.2  memcpy to the buf
        //        3. Eigen::Tensor buffer :
        //          3.1 grab buf in RowMajor/ColMajor layout: tensor.data();
        //          3.2 convert using Eigen::TensorMap : Eigen::TensorMap<Eigen::Tensor<Type, NUM_DIMS>>(buf)
        //  _molded_images_mapped = Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>>(&data[0], 1, molded_shape_H, molded_shape_W, 3);
        for (int h=0; h < molded_shape(1); h++) {
            for (int w=0; w < molded_shape(2); w++) {
                _molded_images_mapped(0, h, w, 0) = molded_images(0, h, w, 2);
                _molded_images_mapped(0, h, w, 1) = molded_images(0, h, w, 1);
                _molded_images_mapped(0, h, w, 2) = molded_images(0, h, w, 0);
            }
        }
        LOG(INFO) << "_molded_images_mapped(0,0,0,:3): " << _molded_images_mapped(0, 0, 0, 0) << " "
                  << _molded_images_mapped(0, 0, 0, 1) << ' '
                  << _molded_images_mapped(0, 0, 0, 2);
        inputs->emplace_back("input_image", _molded_images);

        tfe::Tensor _images_metas(tf::DT_FLOAT, tf::TensorShape({1, images_metas.cols() } ) );
        auto _images_metas_mapped = _images_metas.tensor<float, 2>();
        for (int i=0; i < images_metas.cols(); i++)
        {
            _images_metas_mapped(0, i) = images_metas(0, i);
        }
        inputs->emplace_back("input_image_meta", _images_metas);

        tfe::Tensor _anchors(tf::DT_FLOAT, tf::TensorShape({1, anchors.rows(), anchors.cols()}));
        auto _anchors_mapped = _anchors.tensor<float, 3>();
        for (int i=0; i < anchors.rows(); i++)
        {
            for (int j=0; j < anchors.cols(); j++)
            {
                 _anchors_mapped(0,i,j) = anchors(i,j);
            }
        }
        inputs->emplace_back("input_anchors", _anchors);

        // @todo : TODO
        // run base_engine_ detection
        // see examples from main.cpp, usage of TensorFlowEngine

        // load saved_model.pb
//      tfe::FutureType fut = base_engine_->Run(*inputs, *outputs,
//                                              {"mrcnn_detection/Reshape_1:0", "mrcnn_class/Reshape_1:0", "mrcnn_bbox/Reshape:0", "mrcnn_mask/Reshape_1:0", "ROI/packed_2:0", "rpn_class/concat:0", "rpn_bbox/concat:0"}, {});
//        // load mrcnn.pb
//      tfe::FutureType fut = base_engine_->Run(*inputs, *outputs,
//                                              {"output_detections:0", "output_mrcnn_class:0", "output_mrcnn_bbox:0", "output_mrcnn_mask:0", "output_rois:0", "output_rpn_class:0", "output_rpn_bbox:0"}, {});
//        // load mrcnn_tmp.pb
        tfe::FutureType fut = base_engine_->Run(*inputs, *outputs,
                                                {"mrcnn_detection/Reshape_1:0", "mrcnn_class/Reshape_1:0", "mrcnn_bbox/Reshape:0", "mrcnn_mask/Reshape_1:0", "ROI/packed_2:0", "rpn_class/concat:0", "rpn_bbox/concat:0"}, {});

        // pass fut object to anther thread by value to avoid undefined behaviors
        std::shared_future<tfe::ReturnType>  fut_ref( std::move(fut) );

        // wrap fut with a new future object and pass local variables in
        std::future<ReturnType> wrapped_fut = std::async(std::launch::async, [=, &rets]() -> ReturnType {
            LOG(INFO) << "enter into sfe TF handler ...";

            // fetch result
            fut_ref.wait();

            tf::Status status = fut_ref.get();
            std::string graph_def = base_model_dir_;
            if (status.ok()) {

                if (outputs->size() == 0) {
                    LOG(INFO) << format("[Main] Found no output: %s!", graph_def.c_str(), status.ToString().c_str());
                    return status;
                }
                LOG(INFO) << format("[Main] Success: infer through <%s>!", graph_def.c_str());
                // @todo : TODO fill out the detectron result

		// set results to rets
		...
                }

            } else {
                LOG(INFO) << format("[Main] Failed to infer through <%s>: %s!", graph_def.c_str(), status.ToString().c_str());
            }
            return status;
        });
	
	return wrapped_fut;

```

One of key operations involved is to convert an object of type Eigen::Tensor to that tf::Tensor vice veras. Another one is the conversion between type of
Eigen::Tensor and Eigen::Matrix.

We want to understand that internally tensorflow uses Eigen::Tensor to distribute linear algebra computation and "tf::Tensor" does not support subscrition. To operate on "tf::Tensor", 
we have to map the internal buffer to Eigen::Tensor first, i.e. Eigen::TensorMap which is almost identical to Eigen::Tensor.

Another thing is recommended to keep in mind is that operations among objects of type Eigen::Tensor are for building express tree. Eigen::Tensor is handy for high dimension \(>=4\) operations and Eigen::Matrix is better at low dimensions \(<=3\)

## Eigen::Tensor to Tensorflow::Tensor

Eigen::Tensor is easy for high dimension data manipulation : accessing, writing, and subscription. To operate "Tensorflow::Tensor" we first need to convert data of "tf::Tensor" to Eigen::TensorMap in row major layout.

In the above example:

```c++
auto _molded_images_mapped = _molded_images.tensor<float, 4>();
/*
 * or use tensorflow::Tensor api:
 * float* buf = _molded_images_mapped.flat<float>().data();
 */ 
float* buf = _molded_images_mapped.data(); 

```

we can assign the tf tensor by modifying "_molded_images_mapped". Alternatively, as discussed in [issue#8033](https://github.com/tensorflow/tensorflow/issues/8033) we can directly "memcpy" data to the buffer.

> std::copy(another_buf, another_buf + H*W*CHANNELS, buf); // copy another_buf to tf Tensor _molded_images_mapped

## Eigen::Tensor to Eigen::Matrix

Using the same technique we can quickly convert Eigen::Tensor to Eigen::Matrix, here I provided a simple wrapper to do the job:

```c++
// extract mask
Eigen::MatrixX<float> numeric_mat;

Eigen::Tensor<float, 3, Eigen::RowMajor> mask_slices = std::move( mask.chip(i, 0) );
Eigen::Tensor<float, 2, Eigen::RowMajor> _mask = std::move( mask_slices.chip(class_id, 2) );
// move Eigen::Tensor _mask (extacted from output of type tf::Tensor)
eigen_tensor_2_matrix<float, 2>(_mask, numeric_mat, mask.dimension(1), mask.dimension(2));

```



