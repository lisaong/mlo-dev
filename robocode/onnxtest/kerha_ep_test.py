#!/usr/bin/env python3
import os
import sys
sys.path = ['.'] + sys.path

import onnxruntime as onnxrt
import onnx
import numpy as np
from helper import get_name
import subprocess
import pathlib
import robopy.hat
import json
import itertools
from timeit import default_timer as timer
import pandas as pd

# onnxrt.set_default_logger_severity(1)
rng = np.random.default_rng(seed=9876)

# Keep this True when committing to tree
PERF_TESTING_MODE = True

def download(url, path, overwrite=False):
    from six.moves import urllib
    if os.path.exists(path) and not overwrite:
        return
    print('Downloading {} to {}.'.format(url, path))
    urllib.request.urlretrieve(url, path)


def get_cat_image():
    from PIL import Image
    url = 'https://gist.githubusercontent.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/fa7ef0e9c9a5daea686d6473a62aacd1a5885849/cat.png'
    dst = 'cat.png'
    real_dst = os.path.abspath(os.path.join(os.path.dirname(__file__), dst))
    download(url, real_dst)
    img = Image.open(real_dst).resize((224, 224))
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    return np.asarray(img.astype('float32'))


def make_matmul_1024_model():
    return make_matmul_model(1024, 1024, 1024, 'matmul_1024.onnx')


def make_gpt2_model(batch, seq):
    base_name = "gpt2_small"
    opt_suffix = "_opt"
    specifier = f"_b{batch}_s{seq}"
    expected_name = f"{base_name}{specifier}{opt_suffix}.onnx"
    try:
        return get_name(expected_name)
    except FileNotFoundError:
        pass

    gpt = get_name(f"{base_name}.onnx")
    model = onnx.load(gpt)

    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = batch
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = seq

    temp_name = f"{base_name}{specifier}"
    onnx.save(model, temp_name)

    sess_option = onnxrt.SessionOptions()
    sess_option.intra_op_num_threads = 1
    sess_option.inter_op_num_threads = 1
    sess_option.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    sess_option.optimized_model_filepath = pathlib.Path(temp_name).with_name(expected_name).as_posix()
    sess_option.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    _ = onnxrt.InferenceSession(temp_name, sess_option, providers=['CPUExecutionProvider'])
    return get_name(sess_option.optimized_model_filepath)


def make_fused_matmul_model(M, N, K, batch=[], alpha=1.0, transA=0, transB=0, filename=None, overwrite=False):
    specifier = '_'.join([str(x)
                          for x in ['', M, N, K, alpha, transA, transB]])
    expected_name = filename or f'fused_matmul_{specifier}.onnx'

    def make_model():
        import onnx
        from onnx import helper, TensorProto

        def trans_no_trans(val: int, trans: bool):
            trans_val = -1 if val == -2 else -2
            return trans_val if trans else val

        A_shape = [M, K]
        B_shape = [K, N]
        real_A_shape = [A_shape[trans_no_trans(-2, transA)], A_shape[trans_no_trans(-1, transA)]]
        real_B_shape = [B_shape[trans_no_trans(-2, transB)], B_shape[trans_no_trans(-1, transB)]]

        graph = helper.make_graph(
            [  # nodes
                helper.make_node("FusedMatMul", ["A", "B"], [
                                 "C"], f"MatMul_{specifier}",
                                 domain="com.microsoft",
                                 alpha=alpha, transA=transA, transB=transB),
            ],
            f"fused_matmul_{specifier}",  # name
            [  # inputs
                helper.make_tensor_value_info('A', TensorProto.FLOAT, batch + real_A_shape),

                # Order is transposed
                helper.make_tensor_value_info('B', TensorProto.FLOAT, batch + real_B_shape),
            ],
            [  # outputs
                helper.make_tensor_value_info('C', TensorProto.FLOAT, batch + [M, N]),
            ],
            [  # initializers
            ])
        model = helper.make_model(graph)
        onnx.save(model, 'testdata/' + expected_name)
        return get_name(expected_name)

    if overwrite:
        return make_model()

    try:
        model = get_name(expected_name)
        return model
    except FileNotFoundError:
        return make_model()



def make_matmul_model(M, N, K, batch=[], filename=None, overwrite=False):
    specifier = 'x'.join(map(str, batch)) + '_'.join(map(str, ['', M, N, K]))
    expected_name = filename or f'matmul_{specifier}.onnx'

    def make_model():
        import onnx
        from onnx import helper, TensorProto

        graph = helper.make_graph(
            [  # nodes
                helper.make_node("MatMul", ["A", "B"], [
                                 "C"], f"MatMul_{specifier}"),
            ],
            f"matmul_{specifier}",  # name
            [  # inputs
                helper.make_tensor_value_info(
                    'A', TensorProto.FLOAT, batch + [M, K]),

                # Order is transposed
                helper.make_tensor_value_info(
                    'B', TensorProto.FLOAT, batch + [K, N]),
            ],
            [  # outputs
                helper.make_tensor_value_info(
                    'C', TensorProto.FLOAT, batch + [M, N]),
            ],
            [  # initializers
            ])
        model = helper.make_model(graph)
        onnx.save(model, 'testdata/' + expected_name)
        return get_name(expected_name)

    if overwrite:
        return make_model()

    try:
        model = get_name(expected_name)
        return model
    except FileNotFoundError:
        return make_model()


def make_gemm_model(M, N, K, alpha=1.0, beta=1.0, transA=0, transB=0, filename=None, overwrite=False, use_constant_weights=True):
    suffix = '_'.join([str(x) for x in ['', M, N, K, alpha, beta, transA, transB]])
    expected_name = filename or f'gemm{suffix}.onnx'

    def make_model():
        import onnx
        from onnx import helper, TensorProto

        def trans_no_trans(val: int, trans: bool):
            trans_val = -1 if val == -2 else -2
            return trans_val if trans else val

        A_shape = [M, K]
        B_shape = [K, N]
        real_A_shape = [A_shape[trans_no_trans(-2, transA)], A_shape[trans_no_trans(-1, transA)]]
        real_B_shape = [B_shape[trans_no_trans(-2, transB)], B_shape[trans_no_trans(-1, transB)]]
        real_C_shape = [N]
        real_Y_shape = [M, N]

        initializers = []
        if use_constant_weights:
            b_tensor = helper.make_tensor(
                "B", TensorProto.FLOAT, real_B_shape, np.random.random(real_B_shape).astype(dtype=np.float32).flatten())
            initializers.append(b_tensor)

            # c_tensor = helper.make_tensor(
            #     "C", TensorProto.FLOAT, real_C_shape, np.random.random(real_C_shape).astype(dtype=np.float32).flatten())
            # initializers.append(c_tensor)

            inputs = [
                helper.make_tensor_value_info('A', TensorProto.FLOAT, real_A_shape),

                helper.make_tensor_value_info('C', TensorProto.FLOAT, real_C_shape),
            ]
        else:
            inputs = [
                helper.make_tensor_value_info('A', TensorProto.FLOAT, real_A_shape),
                helper.make_tensor_value_info('B', TensorProto.FLOAT, real_B_shape),
                helper.make_tensor_value_info('C', TensorProto.FLOAT, real_C_shape),
            ]

        graph = helper.make_graph(
            [  # nodes
                helper.make_node("Gemm", ["A", "B", "C"], [
                                 "Y"], f"Gemm{suffix}",
                                 alpha=alpha, beta=beta, transA=transA,
                                 transB=transB),
            ],
            f"gemm{suffix}",  # name
            inputs,
            [  # outputs
                helper.make_tensor_value_info('Y', TensorProto.FLOAT, real_Y_shape),
            ],
            initializers
            )
        model = helper.make_model(graph)
        onnx.save(model, 'testdata/' + expected_name)
        return get_name(expected_name)

    if overwrite:
        return make_model()

    try:
        model = get_name(expected_name)
        return model
    except FileNotFoundError:
        return make_model()


def make_providers(robocode_settings):
    return [
        ('RoboCodeExecutionProvider', {
            'robocode_settings': robocode_settings
        }),
        ('CPUExecutionProvider', {}),
    ]


def _get_os_name():
    if sys.platform.startswith('win'):
        return 'windows'
    elif sys.platform.startswith('darwin'):
        return 'macos'
    else:
        return 'linux'


def _is_windows():
    return _get_os_name() == 'windows'


def _create_dllmain(filename: str):
    f = pathlib.Path(filename).with_suffix(".cc")
    with open(os.path.join(f), 'w') as dllmain_cc:
        print(
            """
            #include <windows.h>
            BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID) { return TRUE; }
            """,
            file=dllmain_cc
        )

    obj_file = f.with_suffix(".obj")
    subprocess.run(['cl', '/Fo' + str(obj_file.absolute()), '/c', str(f.absolute())], check=True)
    return str(obj_file.absolute())


def robopy_provider_settings(hat_package_dir):

    # TODO: HAT packages should have an entry for the base
    hat_dir = pathlib.Path(hat_package_dir)
    package_name = hat_dir.name

    from robopy.hat import ONNXHATPackage
    hat = ONNXHATPackage(hat_dir)
    found_functions = hat.get_functions_for_target(
        os=_get_os_name(), arch='x86_64')

    print("Found the following functions:")
    for fn in found_functions:
        print(f"\t{fn.name}")

    objs = set([str(pathlib.Path(func.link_file).absolute()) for func in
                found_functions if hasattr(func, 'link_file')])
    so_file = (
        hat_dir / package_name).with_suffix(".dll" if _is_windows() else ".so")
    so_file = str(so_file.absolute())

    if _is_windows():
        objs.add(_create_dllmain(
            (hat_dir / package_name).with_name(package_name + "_dllmain.cc")))

        subprocess.run(
            ['link', '-dll', '-FORCE:MULTIPLE'] +
            [f'-EXPORT:{fn.name}' for fn in found_functions] +
            [f'-out:{so_file}'] + list(objs), check=True)

    else:
        subprocess.run(
            ['g++',
             '-shared',
             '-fPIC',
             '-g',
             '-o', so_file
             ] + list(objs), check=True)
    settings = {}
    settings['custom_library'] = so_file
    # node_to_func = dict([
    #     (node_type, found_function.function.auxiliary)
    #     for node_type, function_list in found_functions.items()
    #     for found_function in function_list
    # ])
    # for node_type, function_list in found_functions.items():
    #     aux = fn.auxiliary
    #     onnx = aux['onnx']
    #     node_to_func[onnx['node']] = {
    #         'func_name': fn.name,
    #         'node_args': onnx['node_args']
    #     }

    node_to_func = {}
    for func in found_functions:
        if not func.onnx: continue

        node_funcs = node_to_func.setdefault(func.onnx[ONNXHATPackage.NodeTypeKey], [])
        node_funcs.append(func.onnx)
    settings['node_to_func'] = node_to_func

    return settings


def robopy_provider(hat_package_dir):
    settings = robopy_provider_settings(hat_package_dir)
    return make_providers(json.dumps(settings))

# def test1():
#     sess = onnxrt.InferenceSession(get_name(
#         "mul_1.onnx"), providers=make_providers("generic robocode settings string"))

#     x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
#     input_name = sess.get_inputs()[0].name
#     output_name = sess.get_outputs()[0].name
#     res = sess.run([output_name], {input_name: x})
#     output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
#     np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)


# def test2():
#     sess = onnxrt.InferenceSession(get_name(
#         "matmul_1.onnx"), providers=make_providers("generic robocode settings string"))
#     x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
#     input_name = sess.get_inputs()[0].name
#     output_name = sess.get_outputs()[0].name
#     res = sess.run([output_name], {input_name: x})
#     output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
#     np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)


def test3():
    m = get_name("matmul_1.onnx")
    from onnxruntime.robopy.model_optimizer import optimize_model
    optimized_lib = optimize_model(
        m, package_name="foobar", output_dir="foobar_out")
    sess = onnxrt.InferenceSession(m, providers=robopy_provider(optimized_lib))
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    res = sess.run([output_name], {input_name: x})
    output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
    np.testing.assert_allclose(
        output_expected, res[0], rtol=1e-05, atol=1e-08, verbose=True)


def test4():
    m = get_name("matmul_1.onnx")
    from onnxruntime.robopy.model_optimizer import optimize_model
    optimized_lib = optimize_model(
        m, package_name="foobar", output_dir="foobar_out")
    return optimized_lib


def emit_hat_package_for_model(model, package_name, output_dir):
    # from onnxruntime.robopy.model_optimizer import optimize_model

    from robopy import onnx_emitter

    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)

    inferred_model, optimized_lib = onnx_emitter.emit_package_for_model(model, str(pathlib.Path(output_dir).absolute()))

    return inferred_model, optimized_lib


def create_robocode_settings_from_package(hat_package):
    settings = robopy_provider_settings(hat_package)
    return {'robocode_settings': json.dumps(settings)}


def test5():
    model = make_matmul_1024_model()

    output_dir = "hat_packages"
    # Creates a HAT package with emitted functions for certain nodes in the model
    emit_hat_package_for_model(
        model, package_name="matmul_1024", output_dir=output_dir)

    # HAT package is created into a shared library
    # Setting dictionary is created to map from emitted functions to nodes
    robocode_settings = create_robocode_settings_from_package(output_dir)
    providers = [
        ('RoboCodeExecutionProvider', robocode_settings),
        ('CPUExecutionProvider', {}),
    ]

    sess = onnxrt.InferenceSession(model, providers=providers)

    A = rng.random(size=(1024, 1024), dtype=np.float32)
    B = rng.random(size=(1024, 1024), dtype=np.float32)
    output_name = 'C'
    res = sess.run([output_name],
                   {'A': A,
                    'B': B})

    # Verify
    C = A@B
    try:
        np.testing.assert_allclose(C, res[0], rtol=1e-05, atol=1e-08, verbose=True)
    except AssertionError as e:
        print(e)

    return res


def test6():
    resnet = get_name('resnet18-v2-7.onnx')
    output_dir = "hat_packages"
    emit_hat_package_for_model(resnet, package_name="resnet_18", output_dir=output_dir)

    # HAT package is created into a shared library
    # Setting dictionary is created to map from emitted functions to nodes
    robocode_settings = create_robocode_settings_from_package(output_dir)
    providers = [
        ('RoboCodeExecutionProvider', robocode_settings),
        ('CPUExecutionProvider', {}),
    ]

    sess = onnxrt.InferenceSession(resnet, providers=providers)

    output_name = 'resnetv22_dense0_fwd'
    res = sess.run([output_name],
                   {'data': get_cat_image()})

    return res


def test6_cpu_only():
    resnet = get_name('resnet18-v2-7.onnx')
    # output_dir = "hat_packages"
    # emit_hat_package_for_model(resnet, package_name="resnet_18", output_dir=output_dir)

    # HAT package is created into a shared library
    # Setting dictionary is created to map from emitted functions to nodes
    # robocode_settings = create_robocode_settings_from_package(output_dir)
    providers = [
        # ('RoboCodeExecutionProvider', robocode_settings),
        ('CPUExecutionProvider', {}),
    ]

    sess = onnxrt.InferenceSession(resnet, providers=providers)

    output_name = 'resnetv22_dense0_fwd'
    res = sess.run([output_name],
                   {'data': get_cat_image()})

    return res


def test7():
    # A_shape = [1, 512] # M, K
    # B_shape = [1000, 512] # N, K
    # C_shape = [1, 1000] # M, N

    M, N, K = [1, 1000, 512]

    # M, N, K = [512, 512, 256]
    model = make_matmul_model(M=M, N=N, K=K, overwrite=True)

    output_dir = f"hat_packages/matmul_{M}_{N}_{K}"
    # Creates a HAT package with emitted functions for certain nodes in the model
    emit_hat_package_for_model(
        model, package_name=f"matmul_{M}_{N}_{K}", output_dir=output_dir)

    # HAT package is created into a shared library
    # Setting dictionary is created to map from emitted functions to nodes
    robocode_settings = create_robocode_settings_from_package(output_dir)
    providers = [
        ('RoboCodeExecutionProvider', robocode_settings),
        ('CPUExecutionProvider', {}),
    ]

    sess = onnxrt.InferenceSession(model, providers=providers)

    A = rng.random(size=(M, K), dtype=np.float32)
    B = rng.random(size=(K, N), dtype=np.float32)
    output_name = 'C'
    res = sess.run([output_name],
                   {'A': A,
                    'B': B})

    # Verify
    C = A@B
    try:
        np.testing.assert_allclose(C, res[0], rtol=1e-05, atol=1e-08, verbose=True)
    except AssertionError as e:
        print(e)
    return res


def test8():
    M, N, K = [1, 1000, 512]

    # M, N, K = [512, 512, 256]
    model = make_gemm_model(M=M, N=N, K=K, overwrite=True)

    output_dir = f"hat_packages/gemm_{M}_{N}_{K}"
    # Creates a HAT package with emitted functions for certain nodes in the model
    emit_hat_package_for_model(
        model, package_name=f"gemm_{M}_{N}_{K}", output_dir=output_dir)

    # HAT package is created into a shared library
    # Setting dictionary is created to map from emitted functions to nodes
    robocode_settings = create_robocode_settings_from_package(output_dir)
    providers = [
        ('RoboCodeExecutionProvider', robocode_settings),
        ('CPUExecutionProvider', {}),
    ]

    sess = onnxrt.InferenceSession(model, providers=providers)

    A = rng.random(size=(M, K), dtype=np.float32)
    B = rng.random(size=(K, N), dtype=np.float32)
    C = rng.random(size=(M, N), dtype=np.float32)
    output_name = 'Y'
    res = sess.run([output_name],
                   {'A': A,
                    'B': B, 'C': C})

    # Verify
    Y = A@B + C
    try:
        np.testing.assert_allclose(Y, res[0], rtol=1e-05, atol=1e-08, verbose=True)
    except AssertionError as e:
        print(e)
    return res


def get_min_input_sets(input_shapes, MB_to_exceed=50):
    "cf ELL/ONNXBenchmarks"
    sizes = []
    for shape in input_shapes:
        size = 1
        for s in shape:
            size *= s
        sizes.append(size)
    set_size = np.sum(sizes) * 4  # Size of float tensors in bytes
    return ((MB_to_exceed * 1024 * 1024) // set_size) + 1


def test9():
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    # options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = False

    resnet = get_name('resnet18-v2-7.onnx')
    output_dir = "hat_packages"
    emit_hat_package_for_model(resnet, package_name="resnet_18", output_dir=output_dir)

    output_name = 'resnetv22_dense0_fwd'

    providers = [
        ('CPUExecutionProvider', {}),
    ]

    cpu_sess = onnxrt.InferenceSession(
        resnet, sess_options=options, providers=providers)

    # Create inputs
    input_list = cpu_sess.get_inputs()
    input_shapes = [i.shape for i in input_list]
    num_input_sets = 1 or get_min_input_sets(input_shapes)
    print("\t\tUsing {} input sets".format(num_input_sets))
    input_sets = []
    for i in range(num_input_sets):
        ort_inputs = {}
        for shape, inp in zip(input_shapes, input_list):
            ort_inputs[inp.name] = rng.random(shape).astype(dtype=np.float32)
        input_sets.append(ort_inputs)

    cpu_res = cpu_sess.run([output_name], input_sets[0])

    # HAT package is created into a shared library
    # Setting dictionary is created to map from emitted functions to nodes
    robocode_settings = create_robocode_settings_from_package(output_dir)
    providers = [
        ('RoboCodeExecutionProvider', robocode_settings),
        ('CPUExecutionProvider', {}),
    ]

    rc_sess = onnxrt.InferenceSession(
        resnet,  sess_options=options, providers=providers)
    rc_res = rc_sess.run([output_name], input_sets[0])

    try:
        np.testing.assert_allclose(cpu_res[0], rc_res[0], rtol=1e-05, atol=1e-08, verbose=True)
    except AssertionError as e:
        print(e)
    return (cpu_res[0], rc_res[0])

def get_input_sets(ort_session, num_additional=10):
    input_shapes = []
    for inp in ort_session.get_inputs():
        input_shapes.append(inp.shape)

    generator = np.random.default_rng(seed=2021)

    # Create inputs
    input_list = ort_session.get_inputs()
    num_input_sets = 1 if not PERF_TESTING_MODE else get_min_input_sets(input_shapes) + num_additional
    print("\t\tUsing {} input sets".format(num_input_sets))
    input_sets = []
    for i in range(num_input_sets):
        ort_inputs = {}
        for i, inp in enumerate(input_list):
            if 'int64' in inp.type:
                ort_inputs[inp.name] = generator.integers(0, 255, size=input_shapes[i], dtype=np.int64)
            else:
                ort_inputs[inp.name] = generator.random(input_shapes[i]).astype(dtype=np.float32)
        input_sets.append(ort_inputs)
    return input_sets


def test_model_with_robocode(model, output_dir, outputs=None, options=None, num_results=10000, syms=None):
    os.environ["OMP_NUM_THREADS"] = "1"

    model_name = pathlib.Path(model).name
    output_dir = pathlib.Path(output_dir) / model_name
    #model_name, emitted_lib = emit_hat_package_for_model(
    #    model, package_name=model_name, output_dir=output_dir)

    #emit_hat_package_for_model(model, package_name=model_name, output_dir=output_dir)

    providers = [
        ('CPUExecutionProvider', {}),
    ]

    cpu_sess = onnxrt.InferenceSession(
        model, sess_options=options, providers=providers)

    # Create inputs
    input_sets = get_input_sets(cpu_sess)
    num_input_sets = len(input_sets)

    # HAT package is created into a shared library
    # Setting dictionary is created to map from emitted functions to nodes
    robocode_settings = create_robocode_settings_from_package(output_dir)
    providers = [
        ('RoboCodeExecutionProvider', robocode_settings),
        ('CPUExecutionProvider', {}),
    ]

    rc_sess = onnxrt.InferenceSession(
        model,  sess_options=options, providers=providers)

    cpu_res = cpu_sess.run(outputs, input_sets[0])
    rc_results = rc_sess.run(outputs, input_sets[0])

    try:
        np.testing.assert_allclose(
            cpu_res[0], rc_results[0], rtol=1e-03, atol=1e-05, verbose=True)
    except AssertionError as e:
        print("!" * 20)
        print("Value mismatch detected")

        # Don't interrupt perf testing mode
        if not PERF_TESTING_MODE:
            raise e
        else:
            print(e)

    _ = input("Press any key to continue (CPU)")
    print("*" * 20)
    print(f"Starting timing for iterations={num_results}")
    start_time = timer()
    for i in range(num_results):
        cpu_res = cpu_sess.run(outputs, input_sets[i % num_input_sets])
    end_time = timer()
    cpu_sess.end_profiling()

    cpu_time = ((end_time - start_time) * 1000) / num_results

    _ = input("Press any key to continue (HAT)")
    start_time = timer()
    for i in range(num_results):
        rc_results = rc_sess.run(outputs, input_sets[i % num_input_sets])
    end_time = timer()
    rc_sess.end_profiling()

    rc_time = ((end_time - start_time) * 1000) / num_results

    _ = input("Press any key to continue (END)")

    print(f"\n\n{'*' * 10}\nTesting {model_name} with {num_results} iterations\nCPU Time: {cpu_time}\tHAT Time: {rc_time}\n{'*' * 10}\n")
    os.environ.pop("OMP_NUM_THREADS")

    return (cpu_time, rc_time)


def test10():
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = False

    resnet = get_name('resnet18-v2-7.onnx')
    output_dir = "hat_packages"

    return test_model_with_robocode(resnet, output_dir, options=options, num_results=1)


def test11(batch=[]):
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True

    model = make_matmul_model(1024, 1024, 1024, batch=batch)
    output_dir = "hat_packages"

    return test_model_with_robocode(model, output_dir, options=options, num_results=100)


def gemm_single_node_model_test():
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True

    for alpha in [0.0, 0.2, 1.0]:
        for beta in [0.0, 0.4, 1.0]:
            for M, N, K in itertools.combinations_with_replacement([1, 512, 1000], 3):
                for transA in [1, 0]:
                    for transB in [1, 0]:
                        model = make_gemm_model(
                            M, N, K, alpha=alpha, beta=beta, transA=transA, transB=transB)
                        output_dir = "hat_packages"

                        test_model_with_robocode(
                            model, output_dir, options=options, num_results=100)


def gemm_512_512_512_00_00_1_1():
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True

    for alpha in [0.0]:
        for beta in [0.0]:
            for M, N, K in [[512,512,512]]:
                for transA in [1]:
                    for transB in [1]:
                        model = make_gemm_model(
                            M, N, K, alpha=alpha, beta=beta, transA=transA, transB=transB)
                        output_dir = "hat_packages"

                        test_model_with_robocode(
                            model, output_dir, options=options, num_results=100)


def gemm_512_512_512_x_x_1_1():
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True

    for alpha in [1.0, 0.0]:
        for beta in [1.0, 0.0]:
            for M, N, K in [[512,512,512]]:
                for transA in [1]:
                    for transB in [1]:
                        model = make_gemm_model(
                            M, N, K, alpha=alpha, beta=beta, transA=transA, transB=transB)
                        output_dir = "hat_packages"

                        test_model_with_robocode(
                            model, output_dir, options=options, num_results=100)


def gemm_16_16_16_x_x_1_1():
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True

    for alpha in [1.0, 0.0]:
        for beta in [1.0, 0.0]:
            for M, N, K in [[16,16,16]]:
                for transA in [1]:
                    for transB in [1]:
                        model = make_gemm_model(
                            M, N, K, alpha=alpha, beta=beta, transA=transA, transB=transB)
                        output_dir = "hat_packages"

                        test_model_with_robocode(
                            model, output_dir, options=options, num_results=100)


def gemm_4_4_4_x_x_1_1():
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True

    for alpha in [1.0, 0.0]:
        for beta in [1.0, 0.0]:
            for M, N, K in [[4,4,4]]:
                for transA in [1]:
                    for transB in [1]:
                        model = make_gemm_model(
                            M, N, K, alpha=alpha, beta=beta, transA=transA, transB=transB)
                        output_dir = "hat_packages"

                        test_model_with_robocode(
                            model, output_dir, options=options, num_results=100)


def gemm_2_2_2_x_x_1_1():
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True

    for alpha in [0.0]:
        for beta in [1.0]:
            for M, N, K in [[2, 2, 2]]:
                for transA in [1]:
                    for transB in [1]:
                        model = make_gemm_model(
                            M, N, K, alpha=alpha, beta=beta, transA=transA, transB=transB)
                        output_dir = "hat_packages"

                        test_model_with_robocode(
                            model, output_dir, options=options, num_results=100)



# test1()
# test2()
# test3()


# res_5 = test5()

# res_7 = test7()

# res_8 = test8()

# res_6 = test6()
# res_6_cpu_only = test6_cpu_only()

# try:
#     np.testing.assert_allclose(res_6_cpu_only, res_6, rtol=1e-1, atol=1e-02, verbose=True)
# except AssertionError as e:
#     print(e)


# assert len(res_6) == len(res_6_cpu_only)
# assert len(res_6[0]) == len(res_6_cpu_only[0])
# assert len(res_6[0][0]) == len(res_6_cpu_only[0][0])
# assert res_6[0][0][:10] == res_6_cpu_only[0][0][:10]


# test9()

# test10()
# test11()


# gemm_single_node_model_test()


def test13():
    for layout in itertools.combinations_with_replacement([64, 128,256, 512,1024,2048,4096],3):
        M,N,K = layout
        options = onnxrt.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = True


        model = make_matmul_model(M,N,K)
        options.profile_file_prefix = os.path.basename(model)
        output_dir = "hat_packages"

        test_model_with_robocode(model, output_dir, options=options)


def test14():
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = False

    gpt_1_10 = make_gpt2_model(batch=1, seq=10)
    output_dir = "hat_packages"

    return test_model_with_robocode(gpt_1_10, output_dir,  # outputs=[output_name],
                                    options=options, num_results=1000)


def test15():
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True

    model = make_matmul_model(1024, 1024, 1024, batch=[1, 12])
    output_dir = "hat_packages"

    return test_model_with_robocode(model, output_dir, options=options, num_results=100)

def test16():
    import onnx
    import onnxruntime as onnxrt
    import onnxruntime.tools.symbolic_shape_infer as shape_infer

    from helper import get_name

    results = {}
    for base_name in [
        "gpt2_small",
        # "gpt2_medium",
        # "gpt2_large",
        # "gpt2_xlarge"
        ]:
        #for batch, seq in [(1, 10), (1, 128), (1,256), (1,1024)]:0
        for batch, seq in [(1, 128)]:

            def make_optimized_gpt_model(base_name, batch, seq):
                opt_suffix = "_opt"
                specifier = f"_b{batch}_s{seq}"
                expected_name = f"{base_name}{specifier}{opt_suffix}.onnx"
                try:
                    return get_name(expected_name)
                except FileNotFoundError:
                    pass

                gpt = get_name(f"testdata/{base_name}/{base_name}.onnx")
                print("Loading ", gpt)
                model = onnx.load(gpt)

                model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = batch
                model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = seq

                temp_name = f"{base_name}{specifier}"
                #onnx.save_model(model, temp_name, save_as_external_data=True)
                # onnx.external_data_helper.convert_model_to_external_data(model)
                onnx.save_model(model, temp_name)

                sess_option = onnxrt.SessionOptions()
                sess_option.intra_op_num_threads = 1
                sess_option.inter_op_num_threads = 1
                sess_option.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
                sess_option.optimized_model_filepath = expected_name
                sess_option.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
                _ = onnxrt.InferenceSession(temp_name, sess_option, providers=['CPUExecutionProvider'])

                return get_name(sess_option.optimized_model_filepath)

            from onnxruntime.transformers import shape_infer_helper
            optimized_onnx_model = make_optimized_gpt_model(base_name, batch, seq)

            options = onnxrt.SessionOptions()
            options.add_free_dimension_override_by_name("batch", batch)
            options.add_free_dimension_override_by_name("sequence", seq)
            options.add_free_dimension_override_by_name("seq", seq)

            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_profiling = False

            output_dir = "hat_packages"

            num_results = 100
            if seq > 128:
                num_results = 10

            cpu_time, rc_time = test_model_with_robocode(optimized_onnx_model, output_dir, options=options, num_results=num_results)
            name = f"{base_name} batch={batch} seq={seq}"

            results[name] = {
                "ort": cpu_time, "hat_ep": rc_time,
                "ort/hat_ep": cpu_time / rc_time,
            }
            df = pd.DataFrame.from_dict(results, orient='index')

    print(df)
    df.to_csv("test16.csv")


def test17():

    results = {}
    for batch, layout in [([], (10,768,768)),
                          ([], (10,3702,768)),
                          ([], (10,768,3702)),
                          ([], (10,768,2304)),
                          ([1,12], (10,10,64)),
                          ([1,12], (10,64,10)),
                          ]:

        M, N, K = layout
        options = onnxrt.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = False


        model = make_matmul_model(M,N,K, batch=batch)
        options.profile_file_prefix = os.path.basename(model)
        output_dir = "hat_packages"

        cpu_time, rc_time = test_model_with_robocode(model, output_dir, options=options, num_results=100)

        name = "mat_mul batch={} m,n,k={}".format(batch, layout)
        results[name] = {
            "ort": cpu_time, "hat_ep": rc_time
        }
    df = pd.DataFrame.from_dict(results, orient='index')
    print(df)
    df.to_csv("test17.csv")

def fused_matmul_node_test():
    options = onnxrt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = False

    results = {}
    idx = 0
    for alpha in [1.0, 0.2]:
        for transA in [0]:
            for transB in [0]:
                for M, N, K in [
                    (128, 2304, 768),
                    (128, 768, 768),
                    (128, 768, 3072),
                    (63, 63, 63),
                ]:
                    model = make_fused_matmul_model(M, N, K, alpha=alpha, transA=transA, transB=transB)
                    output_dir = "hat_packages"

                    cpu_time, rc_time = test_model_with_robocode(model, output_dir, options=options, num_results=1000)
                    results[idx] = {
                        "M": M,
                        "N": N,
                        "K": K,
                        "alpha": alpha,
                        "cpu_time": cpu_time,
                        "rc_time": rc_time
                    }
                    idx += 1
    df = pd.DataFrame.from_dict(results, orient='index')
    print(df)
    df.to_csv("fused_matmul_node_test.csv")


def test19():
    results = {}
    idx = 0
    for M, N, K, alpha, beta in [
        (128, 2304, 768, 1.0, 1.0),
        (128, 768, 768, 1.0, 1.0),
        (128, 768, 3072, 1.0, 1.0),
        (128, 2304, 768, 1.0, 0.0),
        (128, 768, 768, 1.0, 0.0),
        (128, 768, 3072, 1.0, 0.0),
        (128, 2304, 768, 0.3, 1.0),
        (128, 768, 768, 0.3, 1.0),
        (128, 768, 3072, 0.3, 1.0),

    ]:

        options = onnxrt.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = False

        model = make_gemm_model(M=M, N=N, K=K, alpha=alpha, beta=beta, overwrite=True, use_constant_weights=True)
        output_dir = "hat_packages"

        cpu_time, rc_time = test_model_with_robocode(model, output_dir, options=options, num_results=1000)
        results[idx] = {
            "M": M,
            "N": N,
            "K": K,
            "alpha": alpha,
            "beta": beta,
            "cpu_time": cpu_time,
            "rc_time": rc_time
        }
        idx += 1
    df = pd.DataFrame.from_dict(results, orient='index')
    print(df)
    df.to_csv("test19.csv")


run = True

if run:
    # gpt2
    # _ = test14()

    # resnet
    # _ = test10()

    # gpt2 optimized inferred

    _ = test16()
    # _, emitted_lib = emit_hat_package_for_model(
    #     _, package_name="gpt2_small", output_dir="hat_packages")

    # fused_matmul_node_test()

    # constant b gemm test
    # test19()

    # matmul test
    # _ = test11()

    # gemm test
    # gemm_single_node_model_test()
    # gemm_512_512_512_00_00_1_1()
    # gemm_512_512_512_x_x_1_1()
    # gemm_16_16_16_x_x_1_1()
    # gemm_4_4_4_x_x_1_1()
    # gemm_2_2_2_x_x_1_1()

    # test17()



pass
