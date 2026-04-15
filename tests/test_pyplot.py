import pytest
from unittest.mock import MagicMock
import numpy as np

import glplot.pyplot as gplt

@pytest.fixture(autouse=True)
def mock_glfw_opengl(mocker):
    # Mock OpenGL and GLFW completely for the pyplot wrapper tests
    mocker.patch('glplot.backend.glfw')
    mocker.patch('glplot.backend.glViewport')
    mocker.patch('glplot.backend.glClearColor')
    mocker.patch('glplot.backend.glEnable')
    mocker.patch('glplot.backend.glGenFramebuffers')
    mocker.patch('glplot.backend.glGenTextures')
    mocker.patch('glplot.backend.glBindTexture')
    mocker.patch('glplot.backend.glTexImage2D')
    mocker.patch('glplot.backend.glTexParameteri')
    mocker.patch('glplot.backend.glBindFramebuffer')
    mocker.patch('glplot.backend.glFramebufferTexture2D')
    mocker.patch('glplot.backend.glBlitFramebuffer')
    mocker.patch('glplot.backend.glCheckFramebufferStatus', return_value=0x8CD5) # GL_FRAMEBUFFER_COMPLETE
    
    # We patch GPULinePlot initialization to avoid shader compilation
    mocker.patch('glplot.backend.GPULinePlot._init_shaders')
    mocker.patch('glplot.backend.GPULinePlot._init_buffers')


def test_figure():
    fig = gplt.figure(title="Test", width=800, height=600)
    assert fig.title == "Test"
    assert fig.width == 800
    assert fig.height == 600

def test_plot_lines():
    a = [1, 2, 3]
    b = [0, 1, 2]
    gplt.plot_lines(a, b, x_range=(0, 10))
    fig = gplt._get_or_create_plot()
    assert fig.N == 3
    assert fig._xrange == (0.0, 10.0)

def test_plot_lines_mismatch():
    with pytest.raises(ValueError):
        gplt.plot_lines([1, 2], [1], x_range=(0, 1))

def test_plot_sequence():
    x = [0, 1, 2]
    y = [0, 1, 4]
    gplt.plot(x, y)
    fig = gplt._get_or_create_plot()
    assert fig.N == 2

def test_plot_sequence_mismatch():
    with pytest.raises(ValueError):
        gplt.plot([1, 2], [1])

def test_set_global_alpha():
    gplt.set_global_alpha(0.5)
    fig = gplt._get_or_create_plot()
    assert fig.global_alpha == 0.5

def test_show_and_savefig(mocker):
    mocker.patch('glplot.backend.GPULinePlot.save_current_view')
    gplt.plot([0, 1], [0, 1])
    gplt.show(test_mode=True)
    gplt.savefig("test.png")
    assert gplt._get_or_create_plot()._is_test_mode == True

def test_scatter_api():
    gplt.scatter([0, 1], [0, 1], color=(1, 1, 1, 1), size=10)
    fig = gplt._get_or_create_plot()
    assert len(fig._scatters) == 1

def test_text_api():
    gplt.text(0, 0, "Hello World")
    fig = gplt._get_or_create_plot()
    assert len(fig._text_annotations) == 1

def test_lod_api():
    gplt.set_lod(enabled=True, max_lines_per_px=500)
    fig = gplt._get_or_create_plot()
    assert fig.enable_subsample == True
    assert fig.max_lines_per_px == 500
