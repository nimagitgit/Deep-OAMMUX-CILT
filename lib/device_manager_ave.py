# device_manager.py
# This module manages the SLM and camera devices for capturing images and displaying holograms.
# This module uses old Holoeye SDK to handle a single SLM.
import os
import sys
from sys import exit
from time import sleep
import numpy as np
# Set the absolute path to the DLLs and SDKs for Thorlabs and HOLOEYE
absolute_path_to_dlls = 'C:\\Users\\nima_\\My Drive\\Python\\external_lib\\Thorlab'
os.environ['PATH'] = absolute_path_to_dlls + os.pathsep + os.environ['PATH']
os.add_dll_directory(absolute_path_to_dlls)
sdk_paths = [
    'C:\\Users\\nima_\\My Drive\\Python\\external_lib\\Thorlab',
    'C:\\Program Files\\HOLOEYE Photonics\\SLM Display SDK (Python) v3.2.0\\python'
]
# Add SDK paths to sys.path if they are not already included
for path in sdk_paths:
    if path not in sys.path:
        sys.path.append(path)
        
        
# Import necessary modules from the Thorlab's SDKs
from holoeye import slmdisplaysdk
import thorlabs_tsi_sdk
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK
from thorlabs_tsi_sdk.tl_mono_to_color_enums import COLOR_SPACE
from thorlabs_tsi_sdk.tl_color_enums import FORMAT

class DeviceManager:
    """
    DeviceManager class to manage SLM and camera devices.
    This class initializes the SLM and camera devices, sets up paths for DLLs and SDKs, 
    and provides methods to capture images and display holograms.
    """
    def __init__(self, dll_paths = None, sdk_paths = None):
        self.dll_paths = dll_paths
        self.sdk_paths = sdk_paths
        self.slm = None
        self.camera = None
        self.mono_to_color_processor = None
        self.mono_to_color_sdk = None
        self.camera_sdk = None
        #self.setup_paths()
        
    # Set up paths for DLLs and SDKs
    def setup_paths(self):
        for path in  self.dll_paths:
            print(path)
            os.add_dll_directory(path)
        for path in self.sdk_paths:
            if path not in sys.path:
                sys.path.append(path)
                
    # Set up devices: SLM and camera
    def setup_devices(self,exptime = 4000, totime = 60000):
        self.slm = slmdisplaysdk.SLMInstance()
        if not self.slm.requiresVersion(3):
            print("SDK mismatch")
            #exit(1)
        error = self.slm.open()
        assert error == slmdisplaysdk.ErrorCode.NoError, self.slm.errorString(error)
        

        self.displayOptions = slmdisplaysdk.ShowFlags.PresentAutomatic  # PresentAutomatic == 0 (default)
        starttime = self.slm.timeNow()

        # Initialize the camera instance
        # Set up camera parameters and functions for capturing grayscale images
        self.camera_sdk = TLCameraSDK()
        self.mono_to_color_sdk = MonoToColorProcessorSDK()
        available_cameras = self.camera_sdk.discover_available_cameras()
        if len(available_cameras) < 1:
            raise ValueError("no cameras detected")

        self.camera = self.camera_sdk.open_camera(available_cameras[0]) 
        self.camera.exposure_time_us = int(exptime) # set exposure 
        self.camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
        self.camera.image_poll_timeout_ms = totime  #  second timeout
        self.camera.arm(2)

        self.image_width = self.camera.image_width_pixels
        self.image_height = self.camera.image_height_pixels

        self.mono_to_color_processor = self.mono_to_color_sdk.create_mono_to_color_processor(
        self.camera.camera_sensor_type,
        self.camera.color_filter_array_phase,
        self.camera.get_color_correction_matrix(),
        self.camera.get_default_white_balance_matrix(),
        self.camera.bit_depth)

        self.mono_to_color_processor.color_space = COLOR_SPACE.LINEAR_SRGB  # sRGB color space
        self.mono_to_color_processor.output_format = FORMAT.RGB_PIXEL  # data is returned as sequential RGB values

        self.camera.issue_software_trigger()
        imageFrame = self.camera.get_pending_frame_or_null()

    # Show a sequence of holograms and capture images
    def show_capture(self,hs, slmpause = 500):            
        N = hs.shape[-1]
        imgs_ave = 0.
        Nrep = 3.
        for rep in range(int(Nrep)):
            imgs = []
            for i in range(N):
                #print('i = ', i)
                error = self.slm.showData(hs[0,:,:,i].numpy(), self.displayOptions)
                assert error == slmdisplaysdk.ErrorCode.NoError, self.slm.errorString(error)
                waitDurationMs = slmpause
                waitDurationMeasuredMs = self.slm.timeWaitMs(waitDurationMs)
                #sleep(2)

                imageFrame = self.camera.get_pending_frame_or_null()
                imageFrame = self.camera.get_pending_frame_or_null()

                #print(np.mean(imageFrame.image_buffer))

                color_image_24_bpp = self.mono_to_color_processor.transform_to_24(
                    imageFrame.image_buffer, self.image_width, self.image_height)
                color_image_24_bpp = color_image_24_bpp.reshape(self.image_height,self.image_width,3)
                imageGS = color_image_24_bpp[:,:,1]
                imgs.append(imageGS)
            imgs = np.stack(imgs,-1)
            imgs_ave += imgs/Nrep
        return imgs_ave[None,...]

    # Release the devices
    def release_devices(self):
        try:
            if self.camera:
                self.camera.disarm()
                self.camera.dispose()
        except Exception as e:
            print(f"Error releasing camera: {e}")
            
        try:
            if self.mono_to_color_processor:
                self.mono_to_color_processor.dispose()
        except Exception as e:
            print(f"Error releasing mono to color processor: {e}")
            
        try:
            if self.mono_to_color_sdk:
                self.mono_to_color_sdk.dispose()
        except Exception as e:
            print(f"Error releasing mono to color SDK: {e}")        
        try:
            if self.camera_sdk:
                self.camera_sdk.dispose()
        except Exception as e:
            print(f"Error releasing camera SDK: {e}")
        
        self.slm = None
        print("Rleased")
