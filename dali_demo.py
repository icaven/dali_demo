"""
    Copyright (c) 2024 Ian Cavén
    MIT License
    
    Program to demonstrate the use of the Nvidia DALI package, by transforming video frames from one
    colour space to another using Nvidia CUDA operations.
    
    Reference: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html
    Test videos may be downloaded from https://github.com/NVIDIA/DALI_extra.git
    Can specify the location of the DALI_extra directory using the DALI_EXTRA_PATH environment variable (defaults to ~/Videos).
    
    If the default test video is not found, then a file dialog will be opened to select a video.

"""
import os
import pathlib
import time
from enum import IntEnum, auto
from typing import Union, List, cast

import numpy as np

import nvidia.dali as nd
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import cupy as cp

import dearpygui.dearpygui as dpg

max_batch_size = 1
frame_rate = 60
max_memory_to_use = 2 ** 30
sequence_length = 100       # Will be reduced if the frames are large
maximum_sequence_length = 100

if "DALI_EXTRA_PATH" not in os.environ:
    os.environ["DALI_EXTRA_PATH"] = str(pathlib.Path("~").expanduser() / "Videos" / "DALI_extra")

# Attempt the use a default video; a file selection dialog will be presented if the video isn't found
video_directory = pathlib.Path(os.environ["DALI_EXTRA_PATH"]) / "db" / "video"
test_video = str(video_directory / "sintel" / "video_files" / "sintel_trailer-720p_2.mp4")
# test_video = ""   # Use this to force the file dialog to open during testing


def bt709_to_bt2020_pipeline(input_source_name: str) -> nd.Pipeline:
    """
    Instantiate a pipeline to convert images in the BT.709 colour space to the BT.2020 colour space.
    :param input_source_name:   The name of the input source DataNode key in the run().
    :return: The pipeline.
    """
    python_function_pipe = nd.Pipeline(
            batch_size=1,
            num_threads=1,
            device_id=0,
            exec_async=False,
            exec_pipelined=False,
            seed=42,
    )
    
    # Values from https://en.wikipedia.org/wiki/Rec._709
    bt709_oetf_to_bt709_linear_kernel_code = """
        // Convert to linear using the BT.709 inverse OETF
        output = input < 0.18f ? input / 4.5f : powf((input + 0.099f)/1.099f, 1.0f/0.45f);
    """
    bt709_oetf_to_bt709_linear = cp.ElementwiseKernel(
            "float32 input",
            "float32 output",
            bt709_oetf_to_bt709_linear_kernel_code,
            name="bt709_oetf_to_bt709_linear"
    )
    
    # Values from https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-2-201510-I!!PDF-E.pdf
    bt2020_linear_to_bt2020_oetf_kernel_code = """
        // Apply non-linear OETF curve using the BT.2020 OETF
        const float b = 0.0181f;
        const float a = 1.0993f;    // == 1 + 5.5 * b
        output = input < b ? (input >=  0 ? 4.5f * input : input) :
                 input <= 1.0f ? a * powf(input, 0.45f) - (a - 1.f) : 1.0f;
    """
    
    bt2020_linear_to_bt2020_oetf = cp.ElementwiseKernel(
            "float32 input",
            "float32 output",
            bt2020_linear_to_bt2020_oetf_kernel_code,
            name="bt2020_linear_to_bt2020_oetf"
    )
    
    def bt709_linear_to_bt2020_linear(image: cp.array, matrix_m2_t: cp.array) -> cp.array:
        s = image.shape
        assert s[-1] == 3
        return (image.reshape((-1, 3)) @ matrix_m2_t).reshape(s)
    
    # Matrix M2 from https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2087-0-201510-I!!PDF-E.pdf
    # Rec. ITU-R BT.2087-0 Annex 1
    # Use the transpose of the conversion matrix so that the multiply can be (image @ bt709_to_bt2020_conversion.T).T
    bt709_to_bt2020_conversion = np.array([[0.6274, 0.3293, 0.0433],
                                           [0.0691, 0.9195, 0.0114],
                                           [0.0164, 0.0880, 0.8956]]).astype(np.float32).T
    bt709_to_bt2020_conversion_datanode = nd.types.Constant(bt709_to_bt2020_conversion, device="gpu")
    with python_function_pipe:
        source_image = fn.external_source(name=input_source_name, device='gpu')
        linear_bt_709 = fn.python_function(source_image, device="gpu", function=bt709_oetf_to_bt709_linear)
        linear_bt_2020 = fn.python_function(linear_bt_709, bt709_to_bt2020_conversion_datanode,
                                            device="gpu", function=bt709_linear_to_bt2020_linear)
        oetf_bt_2020 = fn.python_function(linear_bt_2020, device="gpu", function=bt2020_linear_to_bt2020_oetf)
        python_function_pipe.set_outputs(oetf_bt_2020)
    
    return python_function_pipe


@pipeline_def(device_id=0, exec_pipelined=False, prefetch_queue_depth=1, exec_async=False, num_threads=1, batch_size=1)
def hue_adjustment_pipeline(input_source_name: str, hue_adjustment_value_name: str) -> nd.data_node.DataNode:
    """
    Create a pipeline using the pipeline_def decorator to shift the hue of an image.
    :param input_source_name:           The name of the DataNode key for the source image.
    :param hue_adjustment_value_name:   The name of the DataNode key for the hue adjustment value.
    :return: The output data node for the pipeline constructed by the pipeline_def decorator.
    """
    source_image = fn.external_source(name=input_source_name, device='gpu')
    hue_adjustment_value = fn.external_source(name=hue_adjustment_value_name, device='cpu')
    image_with_shifted_hue = fn.hue(source_image, hue=hue_adjustment_value, device='gpu')
    return image_with_shifted_hue


@pipeline_def(batch_size=max_batch_size, num_threads=2, device_id=0)
def video_input_pipeline(filenames: Union[str, List[str]]):
    """ Create a pipeline which reads video files, decodes them and returns images.
    """
    global sequence_length
    pixel_data_type = "float32"
    
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def create_preflight_pipeline() -> nd.data_node.DataNode:
        """
        Create a preflight pipeline to measure the amount of memory used by each frame
        :return: The new pipeline
        """
        preflight_video = fn.readers.video(filenames=filenames, device='gpu', seed=1, sequence_length=1,
                                           dtype=nd.types.to_dali_type(pixel_data_type),
                                           file_list_include_preceding_frame=False,
                                           )
        return preflight_video
    
    preflight_pipe = cast(nd.Pipeline, create_preflight_pipeline())
    preflight_pipe.build()
    all_frames = preflight_pipe.run()
    all_frames = all_frames[0].as_cpu().as_array()
    frame_size = np.prod(all_frames.shape[2:]) * np.dtype(pixel_data_type).itemsize
    max_frames = int(np.floor(max_memory_to_use / frame_size))
    # print(f"Max frames: {max_frames}, frame_size = {frame_size}")
    
    # Now that the maximum number of frames is known, create a reader for the in-memory sequence
    sequence_length = min(max_frames, max(maximum_sequence_length, sequence_length))
    video = fn.readers.video(filenames=filenames, device='gpu', seed=1, sequence_length=sequence_length,
                             dtype=nd.types.to_dali_type(pixel_data_type), file_list_include_preceding_frame=False)
    
    # Scale the values of the video into the 0..1 range, which what the dpg raw texture expects
    video = video * nd.types.Constant(np.float32(1. / 255.))
    return video


# Radio button choices
class ConversionChoices(IntEnum):
    no_conversion = 0
    colour_conversion = auto()
    hue_conversion = auto()
    
    
class ColourTransformer(object):
    """
    This class provides the user interface to the colour transformation.
    """

    def __init__(self):
                
        self.image_sequence_loaded: bool = False
        self.all_frames: Union[None, np.array] = None       # The current sequence of frames
        self.raw_image_data: Union[None, np.array] = None   # The raw float32 data for the current frame
        self.current_frame_number = -1                      # The current frame number in the sequence
        self.direction: int = 1                             # The playback direction (1 = forward, -1 = backward)
        self.batch_size: int = 1                            # Used when training in batches
        self.number_of_frames = 0                           # The number of frames in the sequence
        self.playing = False
        
        self.filename = ""
        self.play_button = None
        self.frame_number_slider = None
        self.texture_registry = None

        self._hue_shift = 0.

        # Identifiers for the GUI controls
        self.image_window_tag = dpg.generate_uuid()
        self.control_window_tag = dpg.generate_uuid()
        self.initial_file_selection_window_tag = dpg.generate_uuid()
        self.texture_tag = dpg.generate_uuid()
        self.texture_registry_tag = dpg.generate_uuid()
        self.image_tag = dpg.generate_uuid()
        self.filename_tag = dpg.generate_uuid()
        self.hue_adjustment_tag = dpg.generate_uuid()
        self.hue_adjustment_value_tag = dpg.generate_uuid()
        self.no_conversion_radio_tag = dpg.generate_uuid()
        self.bt709_bt2020_radio_tag = dpg.generate_uuid()
        self.hue_adjustment_radio_tag = dpg.generate_uuid()
        
        # Pre-built pipelines
        self.conversion_pipe_used: ConversionChoices = ConversionChoices.no_conversion
        self.cached_bt709_to_bt2020_pipeline: nd.Pipeline = bt709_to_bt2020_pipeline("current_input_frame")
        self.cached_bt709_to_bt2020_pipeline.build()
        self.cached_hue_adjustment_pipeline: nd.Pipeline = \
            cast(nd.Pipeline, hue_adjustment_pipeline("current_input_frame", "hue_adjustment_value"))
                                                                
        self.cached_hue_adjustment_pipeline.build()
    
    @property
    def hue_shift(self):
        return self._hue_shift
    
    def set_hue_shift(self, sender):
        value = dpg.get_value(sender)
        self._hue_shift = np.round(value)
            
    def create_texture_for_video(self, full_path_to_video: str):
        """
        Create a texture for the video referenced by the filename.
        :param full_path_to_video:    The full path to the video file
        :return: None
        """
        self.image_sequence_loaded = False
        self.filename = pathlib.Path(full_path_to_video).name
        
        # Set up the input pipeline for decoding the video and run it to load the frames into memory
        pipe = video_input_pipeline(full_path_to_video)
        pipe.build()
        outputs = pipe.run()
        
        # The frames from the first batch of the first output
        self.all_frames = outputs[0].as_cpu().as_array()
        self.batch_size, self.number_of_frames, h, w, c = self.all_frames.shape

        # Initialize the image from the first frame
        self.raw_image_data = self.all_frames[0, 0]
        image_format = dpg.mvFormat_Float_rgba if c == 4 else dpg.mvFormat_Float_rgb
        
        with dpg.texture_registry(tag=self.texture_registry_tag, show=False):
            try:
                dpg.configure_item(self.texture_tag, width=w, height=h, default_value=self.raw_image_data,
                                    format=image_format)
            except SystemError:
                dpg.add_raw_texture(width=w, height=h, default_value=self.raw_image_data,
                                    format=image_format, tag=self.texture_tag)
        
        # Reset the frame number to the beginning and playback in the forward direction
        self.current_frame_number = 0
        self.direction = 1
    
        # Update the frame number slider
        if self.frame_number_slider is not None:
            dpg.configure_item(self.frame_number_slider, max_value=self.number_of_frames - 1)
            dpg.set_value(self.frame_number_slider, self.current_frame_number)
    
    def set_playing(self, value):
        """
        Callback handler for recording the state of playing and the toggling the button text.
        :param value: 
        :return: 
        """
        self.playing = value
        if self.play_button:
            if self.playing:
                dpg.configure_item(self.play_button, label="Pause")
            else:
                dpg.configure_item(self.play_button, label="Play")
    
    def show_windows(self):
        """
        Construct the main window and the control windows.
        :return: 
        """
        def file_selected_callback(sender, app_data):
            if sender in ['file_dialog_id', 'initial', 'file_dialog_id_initial']:
                if pathlib.Path(app_data['file_name']).suffix in ['.mov', '.mp4']:
                    self.image_sequence_loaded = False
                    try:
                        dpg.delete_item(self.image_tag)
                        self.image_tag = dpg.generate_uuid()
                        dpg.delete_item(self.image_window_tag)
                        self.image_window_tag = dpg.generate_uuid()
                        
                        # Create a new texture tag, since the old one can't be deleted (perhaps a bug in dpg?)
                        self.texture_tag = dpg.generate_uuid()
                    except SystemError:
                        pass
                    self.create_texture_for_video(app_data['file_path_name'])
                                        
                    window_bar_height = 20
                    with dpg.window(width=self.raw_image_data.shape[1], 
                                    height=self.raw_image_data.shape[0]+window_bar_height,
                                    no_close=True, no_bring_to_front_on_focus=True,
                                    tag=self.image_window_tag, no_move=True, pos=(0, window_bar_height),
                                    no_title_bar=True):
                        dpg.add_image(label="Image", texture_tag=self.texture_tag, tag=self.image_tag)
                    
                    try:
                        # Update the filename in the controls window
                        dpg.set_value(self.filename_tag, f"{app_data['file_name']}")
                    except SystemError:
                        pass    # Label hasn't been added yet
                    
                    self.image_sequence_loaded = True
                    self.set_playing(True)
                
        def cancel_file_selection_callback(sender, app_data):
            if sender == 'file_dialog_id_initial':
                dpg.stop_dearpygui()
                    
        # Add file selection dialogs to select either a directory or a file (or file sequence)
        def create_file_dialog(tag):
            with dpg.file_dialog(directory_selector=False, file_count=10, show=False, callback=file_selected_callback,
                                 tag=tag, cancel_callback=cancel_file_selection_callback,
                                 width=700, height=400):
                dpg.add_file_extension("Video files (*.mov *.mp4){.mov,.mp4}", color=(0, 255, 255, 255))
                # dpg.add_file_extension("Image files (*.tif[f] *.jpg *.png){.tif,.tiff,.jpg,.png}",
                #                        color=(0, 255, 255, 255))
                
        create_file_dialog("file_dialog_id")
        
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll, enabled_state=True):
                dpg.add_theme_color(dpg.mvThemeCol_Button, value=(23, 140, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, [255, 255, 255])
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)

        dpg.bind_theme(global_theme)
        
        # Start with the initial video, if it exists
        if pathlib.Path(test_video).exists():
            file_selected_callback("initial",
                                   {'file_path_name': test_video, 'file_name': pathlib.Path(test_video).name})
            
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Open...", tag="file_selection_menu_item", callback=lambda: dpg.show_item("file_dialog_id"))
                dpg.add_menu_item(label="Quit", tag="quit_menu_item", callback=dpg.stop_dearpygui)
        
        if self.raw_image_data is not None:
            self.image_sequence_loaded = True
        else:
            # No default video, so present a dialog to select it
            create_file_dialog("file_dialog_id_initial")
            dpg.show_item("file_dialog_id_initial")
            
        with dpg.window(tag=self.control_window_tag, no_close=True, width=300):
            dpg.add_text(f"{self.filename}", tag=self.filename_tag)

            with dpg.group(horizontal=True, horizontal_spacing=5):
                self.play_button = dpg.add_button(label="Play", callback=lambda: self.set_playing(not self.playing))
                dpg.add_text("Frame: ")
                max_frame_number = self.all_frames.shape[1] - 1 if self.image_sequence_loaded else 0
                
                self.frame_number_slider = dpg.add_slider_int(tag="frame_number",
                                                              max_value=max_frame_number,
                                                              min_value=0, callback=self.set_frame_number)
            
            dpg.add_radio_button(("No conversion", "BT.709 -> BT.2020", "Hue shift"),
                                 callback=self.conversion_parameters_changed, horizontal=False)
                        
            with dpg.group(horizontal=True, horizontal_spacing=5):
                dpg.add_slider_float(tag=self.hue_adjustment_tag, min_value=-180, max_value=179,
                                     default_value=0, format="%3.0f", clamped=True, callback=self.set_hue_shift)

        self.set_playing(self.image_sequence_loaded)
                
    def set_frame_number(self, sender):
        frame_number = dpg.get_value(sender)
        self.current_frame_number = frame_number
    
    def conversion_parameters_changed(self, sender, app_data):
        if app_data == 'BT.709 -> BT.2020':
            self.conversion_pipe_used = ConversionChoices.colour_conversion
        elif app_data == "Hue shift":
            self.conversion_pipe_used = ConversionChoices.hue_conversion
        elif app_data == "No conversion":
            self.conversion_pipe_used = ConversionChoices.no_conversion

    def update_image(self):
        """
        Updates the current frame displayed, and possibly performs a colour conversion operation on the image.
        :return: 
        """
        if self.image_sequence_loaded:
            if self.playing:
                # Playback the sequence with a bounce at the start and end
                if (self.direction < 0 and self.current_frame_number == 0) or \
                        (self.direction > 0 and self.current_frame_number == self.number_of_frames - 1):
                    self.direction *= -1
                    
                self.current_frame_number = (self.current_frame_number + self.direction) % self.number_of_frames
                
            # Update the frame number slider
            dpg.set_value(self.frame_number_slider, self.current_frame_number)
            
            current_input_frame = self.all_frames[0, self.current_frame_number, :]
            if self.conversion_pipe_used == ConversionChoices.no_conversion:
                current_frame = current_input_frame
            else:
                image_on_gpu_as_single_frame_batch = cp.array(current_input_frame[np.newaxis, np.newaxis, :])

                if self.conversion_pipe_used == ConversionChoices.hue_conversion:
                    # Get the hue shift and feed it into the pipeline
                    hue_adjustment_value = cp.array((self.hue_shift,), dtype=cp.float32)
                    self.cached_hue_adjustment_pipeline.feed_input(data_node="hue_adjustment_value", data=hue_adjustment_value)
                    output_image = self.cached_hue_adjustment_pipeline.run(current_input_frame=image_on_gpu_as_single_frame_batch)
                else:   # self.conversion_pipe_used == ConversionChoices.colour_conversion:
                    output_image = self.cached_bt709_to_bt2020_pipeline.run(current_input_frame=image_on_gpu_as_single_frame_batch)

                current_frame = output_image[0].as_cpu().as_array().squeeze()

            self.raw_image_data[:] = current_frame


def main():
    dpg.create_context()
    
    # dpg.show_item_registry()      # For debugging
    
    # Create the viewport which contains the application's windows
    dpg.create_viewport(title='Transformation', width=1920, height=1080)
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    ct = ColourTransformer()
    ct.show_windows()
    
    while dpg.is_dearpygui_running():
        ct.update_image()
        dpg.render_dearpygui_frame()
        time.sleep(1./frame_rate)
        
    dpg.destroy_context()


if __name__ == "__main__":
    main()
    