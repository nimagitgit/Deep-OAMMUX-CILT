# device_manager.py
# This module manages the SLM and camera devices for capturing images and displaying holograms.
# This module uses the new Holoeye SDK to handle a multiple (two) SLMs.

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
    #'C:\\Program Files\\HOLOEYE Photonics\\SLM Display SDK (Python) v3.2.0\\python'
    'C:\\Program Files\\HOLOEYE Photonics\\SLM Display SDK (Python) v4.0.0\\api\\python'

]
# Add SDK paths to sys.path if they are not already included
for path in sdk_paths:
    if path not in sys.path:
        sys.path.append(path)
        
        
# Import necessary modules from the Thorlabs and HOLOEYE SDKs
import HEDS
from hedslib.heds_types import *
from hedslib.heds_types import HEDSERR_NoError

# Import necessary modules from the Thorlabs TSI SDKs
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
    def __init__(self, dll_paths= None, sdk_paths= None):
        self.dll_paths = dll_paths
        self.sdk_paths = sdk_paths
        self.slm = None
        self.camera = None
        self.mono_to_color_processor = None
        self.mono_to_color_sdk = None
        self.camera_sdk = None
        self.phase_unit = 2.0*np.pi
        #self.setup_paths()

    # Set up paths for DLLs and SDKs 
    def setup_paths(self):
        for path in  self.dll_paths:
            print(path)
            os.add_dll_directory(path)
        for path in self.sdk_paths:
            if path not in sys.path:
                sys.path.append(path)

    # Load Zernike parameters from a file and apply them to the SLM with the given index.
    def load_zernike(self, index, fpath):
        z_folder = 'C:\\Users\\nima_\\HOLOEYE Photonics\\Pattern Generator\\'
        z_fname = z_folder + 'B_oam_1.zernike.txt'
        err = self.slm[index].setWavelength(532.0)
        err, zernikeParameters = self.slm[index].zernikeLoadParamsFromFile(z_fname)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)
        print(HEDS.SLM.ZernikePrintString(zernikeParameters, "Loaded Zernike parameters from file"))
        print("\nWaiting 3 seconds before applying Zernike parameters ...")
        err = self.slm[index].zernikeApplyParams(zernikeParameters)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

    
    def setup_camera(self, exptime= 4000, totime= 60000):
        """
        Set up the camera for capturing images.
        Initializes the camera, sets exposure time, 
        and prepares functons for capturing grayscale images.
        Args:
            exptime (int): Exposure time in microseconds.
            totime (int): Timeout for image polling in milliseconds.
        """
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

    def setup_slm(self, num= 1, id= None):
        """
        Set up the SLM devices.
        Initializes the SLMs with the specified number and IDs.
        Args:
            num (int): Number of SLMs to initialize.
            id (str or list): ID(s) of the SLM(s) to initialize. 
                              If None, defaults to "index:0", "index:1", etc.
        """
        self.slm = []
        err = HEDS.SDK.Init(4,0)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        for i in range(num):
            if id:
                preselect = id if isinstance(id, str) else id[i]
            else:
                f"index:{i}"
            self.slm.append(HEDS.SLM.Init(preselect = preselect, openPreview = False))
            assert self.slm[i].errorCode() == HEDSERR_NoError, HEDS.SDK.ErrorString(self.slm[i].errorCode())
        self.n_slm = num

    def setup_devices(self, n_slm= 1, id= None, exptime= 4000, totime= 60000):
        """
        Set up both SLM and camera devices.
        Initializes the SLMs and camera with the specified parameters.
        Args:
            n_slm (int): Number of SLMs to initialize.
            id (str or list): ID(s) of the SLM(s) to initialize. 
                              If None, defaults to "index:0", "index:1", etc.
            exptime (int): Exposure time for the camera in microseconds.
            totime (int): Timeout for image polling in milliseconds.
        """
        #id = "index:0"
        #id = "serial:6010-0220"
        #id = "serial:6010-0227"
        if not isinstance(id, list):
            id = [id]
        self.setup_slm(num= n_slm, id= id)
        self.setup_camera(exptime, totime)

    def capture_grey(self):
        """
        Capture a grayscale image from the camera.
        """
        imageFrame = self.camera.get_pending_frame_or_null()
        imageFrame = self.camera.get_pending_frame_or_null()

        color_image_24_bpp = self.mono_to_color_processor.transform_to_24(
            imageFrame.image_buffer, self.image_width, self.image_height)
        color_image_24_bpp = color_image_24_bpp.reshape(self.image_height,self.image_width,3)
        imageGS = color_image_24_bpp[:,:,1]
        return imageGS

    def show(self, holo, index= 0):
        """
        Show a hologram on the specified SLM wit the given index.
        """
        if isinstance(holo, np.ndarray):
            h = holo
        else:
            try:
                h = holo.numpy()
            except:
                print("Wrong input type")

        err = self.slm[index].showPhaseData(data= h, flags= None, phase_unit= self.phase_unit)
        assert err == HEDSERR_NoError, HEDS.SDK.ErrorString(err)             

    def show_capture_multi(self, holos, slmpause= 500):
        """
        Show multiple holograms on the SLMs and capture images.
        Args:
            holos (list or tuple): List or tuple of holograms to be displayed.
            slmpause (int): Pause duration in milliseconds after displaying each hologram.
        Returns:
            np.ndarray: Captured images stacked along the last dimension.
        """
        if not isinstance(holos, list) and not isinstance(holos, tuple):
            holos = [holos]
        N_ch = max(h.shape[-1] for h in holos)
        N_holo = len(holos)         

        if self.n_slm < N_holo:
            raise Exception("Not enough number of SLMs")
        
        for i, holo in enumerate(holos):
            if holo is not None and holo.shape[-1] == 1:
                self.show(holo[0,:,:,0], index= i)
        imgs = []
        for ch in range(N_ch):
            for i, holo in enumerate(holos):
                if holo is None or holo.shape[-1] == 1:
                    continue
                self.show(holo[0,:,:,ch], index= i)

            waitDurationMs = slmpause
            waiterr = self.slm[0].wait(waitDurationMs)
            assert waiterr == HEDSERR_NoError

            imageGS = self.capture_grey()
            imgs.append(imageGS)

        imgs = np.stack(imgs,-1)
        return imgs[None,...]
        
    def show_capture(self, holo, slmpause= 500):
        """
        Show a single hologram on the SLM and capture images.
        Args:
            holo (np.ndarray): Hologram to be displayed.
            slmpause (int): Pause duration in milliseconds after displaying the hologram.
        """
    
        N = holo.shape[-1]
        imgs = []
        for ch in range(N):
            self.show(holo[0,:,:,ch])

            waitDurationMs = slmpause
            waiterr = self.slm[0].wait(waitDurationMs)
            assert waiterr == HEDSERR_NoError

            imageGS = self.capture_grey()
            imgs.append(imageGS)

        imgs = np.stack(imgs,-1)
        return imgs[None,...]

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
