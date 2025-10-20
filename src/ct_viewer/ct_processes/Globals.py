import os
import sys
import cupy as cp
import numpy as np
import pandas
from copy import deepcopy
import colorcet
import matplotlib.pyplot as plt
from matplotlib import cm
from .fast_interp import interp1d, interp2d, interp3d
from cupyx.scipy.interpolate import RegularGridInterpolator, interpn
from cupyx.scipy import ndimage as ndi
import quaternionic as qtn
import configparser


# FileDialog
import datetime
from pathlib import Path
from operator import itemgetter
from contextlib import contextmanager
from typing import Generator, Union
from scipy.io import loadmat, savemat, whosmat
from scipy.io.matlab import matfile_version
import h5py
import gdcm
import nibabel as nib

# All
import dearpygui.dearpygui as dpg

def create_tag(module_prefix: str, 
               dpg_object_designation: str, 
               tag_name: str, 
               separator: str = '|', 
               suffix: str = ''):
    """
    module_prefix <separator> dpg_object_name <separator> tag_name <separator> suffix
    dpg_object, tag_name, suffix = tag.split(separator)
    """
    tag = f'{module_prefix}{separator}{dpg_object_designation}{separator}{tag_name}{separator}{suffix}' if suffix != '' else f'{module_prefix}{separator}{dpg_object_designation}{separator}{tag_name}'
    return tag

class G:
    def __init__(self):
        print('GLOBALS Message: G Imported')

    APP = None
    ROOT = 'root'
    DEBUG_MODE = False
    GPU_MODE = False
    DEVICE = 0
    TEXTURE_TAG = 'main_view_texture'
    TEX_REG_TAG = 'main_texture_registry'
    COLORMAP_TAG = 'app_colormap_registry'
    VALUE_REG_TAG = 'app_value_registry'
    HANDLER_REG_TAG = 'app_mouse_handler_registry'
    ITEM_HANDLER_REG_TAG = 'app_item_handler_registry'
    LOADING_WINDOW_TAG = 'modal_loading_window'
    LOADING_WINDOW_TEXT = 'loading_window_text'
    DATA_SELECTION_WINDOW = 'data_selection_window'
    FILE_INFORMATION_WINDOW = 'file_information_window'
    FILE_INFORMATION_TABLE = 'file_information_table'
    SELECTED_ITEMS_TABLE = 'selected_items_table'
    LANDMARKS_TABLE_TAG = 'landmarks_table'
    TAB_BAR_TAG = 'view_tab_bar'
    VOLUME_TAB_TAG = 'volume_tab'
    ANALYSIS_TAB_TAG = 'analysis_tab'
    LEFT_TAB_BAR = 'left_tab_bar'
    NAVIGATION_TAB_TAG = 'navigation_tab'
    FILE_EXPLORER_TAB_TAG = 'file_explorer_tab'
    GROUP_LAYER_CONTROL_BUTTON = 'group_layer_control_button'
    GROUP_LAYER_RESET_BUTTON = 'group_layer_reset_button'
    DEFAULT_GROUP_LAYER_CONTROL = 'Group'
    MASK_TEXTURE_TAG = 'mask_texture'
    MAX_SELECTABLE_ITEMS = 100
    TEXTURE_IMAGE_TAG = 'Texture_Image'
    TEXTURE_MASK_TAG = 'Mask_Texture_Image'
    TEXTURE_CENTER_TAG = 'Texture_Center'
    MIN_NUMBER_OF_INCREMENTS = 25
    VOLUME_FILE_DIALOG = 'MenuBar_volume_file_dialog'
    SHOW_DEBUG = False
    SHOW_METRICS = False
    SHOW_ITEM_REGISTRY = False

    if GPU_MODE:
        cp.cuda.Device(DEVICE).use()
        print(f'GLOBALS Message: GPU Mode enabled. Device: {DEVICE}, {cp.cuda.Device() = }')
    
    # Themes
    LINE_THEME = 'LINE_THEME'

    # Sliders
    ORIGIN_X_SLIDER = 'origin_x_slider'
    ORIGIN_Y_SLIDER = 'origin_y_slider'

    ### CONFIGURATION ###
    OPERATING_SYSTEM = f'{sys.platform}'

    HOME_PATH_DICT = {'win32': 'HOMEPATH',
                      'linux': 'HOME', 
                      'darwin': 'HOME'}
    
    # USERHOME = Path(os.getenv(HOME_PATH_DICT[OPERATING_SYSTEM]))
    USERHOME = Path.home() 
    CONFIG_DIR = USERHOME.joinpath('.ct_viewer')
    CONFIG_FILE = CONFIG_DIR.joinpath('config.ini')
    CONFIG_PARSER = configparser.ConfigParser(allow_no_value=True)
    DEFAULT_CONFIG_DICT = {'app_settings': {'app_width': 1920,
                                            'app_height': 1080,
                                            'tab_pane_width': 980,
                                            'tab_pane_height': 980,
                                            'main_pane_width': 965,
                                            'main_pane_height': 880,
                                            'main_plot_width': 860,
                                            'main_plot_height': 860,
                                            'main_texture_height': 860,
                                            'main_texture_width': 860,
                                            'colormap_scale_width': 85,
                                            'colormap_scale_height': 860,
                                            'option_panel_width': 450,
                                            'option_panel_height': 600,
                                            'info_box_width': 420,
                                            'info_box_height': 980,
                                            'navigation_window_width': 465,
                                            'navigation_window_height': 980},
                            'directories': {'image_dir': f'{USERHOME}',
                                            'home_path': f'{USERHOME}'},
                            'default_colors': {'default_landmark_color': (113, 237, 235, 255),
                                                'default_crosshair_color': (113, 237, 235, 125)},
                            'plot_settings': {'default_view': 'axial',
                                              'default_cmap': 'Fire'},
                            'texture_settings': {'texture_dimension': 700},
                            'landmark_settings': {'circle_radius': 2},
                            'gpu_settings':{'use_gpu': True, 
                                            'device': 0}
                            }
    

    CONFIG_DICT = dict(DEFAULT_CONFIG_DICT)
    
    CONFIG_PARSER.read_dict(DEFAULT_CONFIG_DICT)

    if not CONFIG_FILE.exists():
        print(f'GLOBALS Message: Creating configuration file at {CONFIG_FILE}.')
        CONFIG_DIR.mkdir(parents = True, exist_ok = True)
        with open(str(CONFIG_FILE), 'w') as config_file:
            CONFIG_PARSER.write(config_file)

    
    else:
        print(f'GLOBALS Message: Reading configuration file at {CONFIG_FILE}')
        
        CONFIG_PARSER.read(CONFIG_FILE)

        for section in CONFIG_PARSER.sections():
            config_section = CONFIG_PARSER[section]
            
            if section not in CONFIG_DICT.keys():
                CONFIG_DICT[section] = {}

            print(f'GLOBALS Message: [{section}]')
            for setting in config_section.keys():
                setting_value = config_section[setting]

                if section == 'app_settings':
                    setting_value = int(setting_value)

                if section == 'default_colors':
                    setting_value = tuple([int(color_ele_str.strip()) for color_ele_str in setting_value[1:-1].split(',')])

                if section == 'texture_settings':
                    setting_value = int(setting_value)

                if section == 'landmark_settings':
                    setting_value = int(setting_value)
                
                CONFIG_DICT[section][setting] = setting_value
                print(f'GLOBALS Message:     {setting} = {setting_value}')


    def save_config(config_version:str ):
        if config_version == 'current': 
            G.CONFIG_PARSER.read_dict(G.CONFIG_DICT)

        elif config_version == 'default': 
            G.CONFIG_PARSER.read_dict(G.DEFAULT_CONFIG_DICT)

        else:
            print('Invalid configuration save option.')
            return
        
        if not G.CONFIG_FILE.parent.exists():
            G.CONFIG_FILE.parent.mkdir(parents = True)
        
        with open(G.CONFIG_FILE, 'w') as config_file:
            G.CONFIG_PARSER.write(config_file)
        
        print('GLOBALS Message: Configuration saved.')

    LOG_DIR = CONFIG_DIR.joinpath('Logs')
    LOG_DIR.mkdir(exist_ok = True)
    # Dictionary for resetting everything to default when closing a volume or set of volumes. 
    GLOBAL_DEFAULTS = {'TEXTURE_DIM': min([CONFIG_DICT['app_settings']['main_texture_height'], 
                                           CONFIG_DICT['app_settings']['main_texture_width']]), # CONFIG_DICT['texture_settings']['texture_dimension'], 
                       'CURRENT_FILE': '', 
                       'FILE_LOADED': False, 
                       'VIEW': CONFIG_DICT['plot_settings']['default_view'], 
                       'VOLUME_KEYS': [], 
                       'N_VOLUMES': 0}
    
    INFORMATION_BOX = 'information_box'
    MAIN_PLOT_VIEW = 'main_plot_view'
    ANALYSIS_PLOT_VIEW = 'analysis_plot_view'
    CURRENT_TAB = 'volume_tab'

    TEXTURE_DIM = min([CONFIG_DICT['app_settings']['main_texture_height'], 
                       CONFIG_DICT['app_settings']['main_texture_width']]) #CONFIG_DICT['texture_settings']['texture_dimension']
    TEXTURE_CENTER = TEXTURE_DIM/2
    
    CURRENT_FILE = ''
    FILE_LOADED = False
    VIEW = GLOBAL_DEFAULTS['VIEW']
    VOLUME_KEYS = []
    
    N_VOLUMES = 0

    sqrt2 = np.sqrt(2)/2

    # These apply to the direction cosine of [1, 0, 0, 0, 1, 0] 
    # and shape like [Rows, Columns, Slices], which corresponds to
    # [X, Y, Z].
    # Which is the standard RCS coordinate system of a CT Scan. 
    QTN_DICT = {'axial': qtn.array([1, 0, 0, 0]),
                'coronal': qtn.array([sqrt2, 0, -sqrt2, 0]),
                'sagittal': qtn.array([sqrt2, -sqrt2, 0, 0])}
    
    NORM_DICT = {'axial': np.array([])}
    
    # {display_name, {name: call_name, module: module, colormap: interp_cmap, colormap_r: interp_cmap_r]}
    COLORMAP_DICT = {'Fire': {'name': 'm_fire', 'module': colorcet},
                     'Gray': {'name': 'm_gray', 'module': colorcet},
                     'Jet': {'name': 'jet', 'module': cm},
                     'Viridis': {'name': 'viridis', 'module': cm},
                     'Black, Green, Yellow': {'name': 'm_kgy', 'module': colorcet},
                     'Blue, Magenta, Yellow': {'name': 'm_bmy', 'module': colorcet}, 
                     'Blues': {'name': 'm_blues', 'module': colorcet},
                     'Blue, Green, Yellow, White': {'name': 'm_bgyw', 'module': colorcet},
                     'Blue, Green, Yellow': {'name': 'm_bgy', 'module': colorcet},
                     'Blue, Black, Red': {'name': 'm_bkr', 'module': colorcet},
                     'Cool Warm': {'name': 'm_coolwarm', 'module': colorcet},
                     'Green, White, Violent': {'name': 'm_gwv', 'module': colorcet},
                     'Blue, Gray, Yellow': {'name': 'm_bjy', 'module': colorcet},
                     'Blue, White, Yellow': {'name': 'm_bwy', 'module': colorcet},
                     'Cyan, White, Red': {'name': 'm_cwr', 'module': colorcet},
                     'Rainbow': {'name': 'm_rainbow', 'module': colorcet},
                     'Colorwheel': {'name': 'm_colorwheel', 'module': colorcet}}
    
    COLORMAP_SCALES = ['Linear',
                       'Absolute',
                       'Log',
                       'Abs-Log']

    def initialize_rgba_interp(colormap_module, colormap_name) -> list[interp1d]:

        print(f'GLOBALS Message: Loading {colormap_name}, {type(getattr(colormap_module, colormap_name))}')
        
        start = 0
        stop = 1
        step = 1/255
        order = 1

        if G.GPU_MODE:
            
            # color_rgba needs to be a list here due to RegularGridInterpolator
            color_rgba = cp.array([getattr(colormap_module, colormap_name)(list(range(256)))])
            # print(f'GLOBALS Message: {color_rgba.shape = }, {color_rgba[:,:,0].T.shape = }, {color_rgba[:,:,0].flatten().shape = }')
            red_interp = RegularGridInterpolator((cp.linspace(0, 1, 256),), color_rgba[:,:,0].flatten(), bounds_error = False, fill_value = None) # (256,)
            green_interp = RegularGridInterpolator((cp.linspace(0, 1, 256),), color_rgba[:,:,1].flatten(), bounds_error = False, fill_value = None)
            blue_interp = RegularGridInterpolator((cp.linspace(0, 1, 256),), color_rgba[:,:,2].flatten(), bounds_error = False, fill_value = None)

            # interp_test = cp.linspace(0, 0.85, 100)
            # print(f'GLOBALS Message: {interp_test.shape = }')
            # print(f'GLOBALS Message: {red_interp(interp_test.T).shape = }, {interp_test.shape = }')

        else:
            color_rgba = getattr(colormap_module, colormap_name)(list(range(256)))
            red_interp = interp1d(start, stop, step, color_rgba[:,0], k = order)
            green_interp = interp1d(start, stop, step, color_rgba[:,1], k = order)
            blue_interp = interp1d(start, stop, step, color_rgba[:,2], k = order)

        return [red_interp, green_interp, blue_interp]
    
    def initialize_colormaps():

        for cmap_key in list(G.COLORMAP_DICT.keys()): # eg: Fire

            cmap_name = f'{G.COLORMAP_DICT[cmap_key]["name"]}' # eg: m_fire
            cmap_name_r = f'{cmap_name}_r'
            
            # G.COLORMAP_DICT[cmap_key][2] is the module the colormap lives in. 
            G.COLORMAP_DICT[cmap_key][f'colormap'] = G.initialize_rgba_interp(G.COLORMAP_DICT[cmap_key]['module'], cmap_name)
            G.COLORMAP_DICT[cmap_key][f'colormap_r'] = G.initialize_rgba_interp(G.COLORMAP_DICT[cmap_key]['module'], cmap_name_r)
        
        default_cmap = G.CONFIG_DICT['plot_settings']['default_cmap']
        
        # Need to set this here so that we call initialize_colormaps first, which is gone in ct_viewer.py
        G.DEFAULT_IMAGE_SETTINGS = {'colormap_combo': default_cmap,
                                    'colormap_scale': f'colormap_{G.COLORMAP_DICT[default_cmap]["name"]}',
                                    'colormap': G.COLORMAP_DICT[default_cmap]['colormap'],
                                    'view': G.CONFIG_DICT['plot_settings']['default_view'],
                                    'min_value': 0,
                                    'max_value': 1200,
                                    'image_size_multiplier': 1}
        
        print(f'GLOBALS Message: Default colormap set to {default_cmap}.')


    # ['Linear', 'Nearest Neighbor', 'Spline Linear', 'Cubic', 'Quintic', 'PCHIP']
    INTERPOLATION_OPTIONS = ['Linear', 'Nearest Neighbor'] #, 'Cubic', 'Quintic']# 'Spline Linear', 'Cubic', 'Quintic', 'PCHIP']
    # INTERPOLATION_DICT = {'Nearest Neighbor': 0,
    #                       'Linear': 1,
    #                       'Cubic': 3,
    #                       'Quintic': 5}
    INTERPOLATION_DICT = {'Nearest Neighbor': 'nearest',
                          'Linear': 'linear'}


    OPTION_TAG_DICT = {'img_index': 'img_index_slider', 
                       'origin_x': 'origin_x_slider', 
                       'origin_y': 'origin_y_slider', 
                       'origin_z': 'origin_z_slider',
                       'norm': 'norm_slider', 
                       'pitch': 'pitch_slider', 
                       'yaw': 'yaw_slider',
                       'roll': 'roll_slider',
                       'pixel_spacing_x': 'pixel_spacing_x_input', 
                       'pixel_spacing_y': 'pixel_spacing_y_input',
                       'slice_thickness': 'slice_thickness_input',
                       
                       'min_intensity': 'min_intensity_slider', 
                       'max_intensity': 'max_intensity_slider', 
                       'colormap_name': 'colormap_combo',
                       'colormap_rescaled': 'colormap_rescaled',
                       'colormap_scale_type': 'colormap_scale_type',
                       'colormap_reversed': 'colormap_reversed'}
    
    
    OPTIONS_DICT = {'img_index_slider': {}, 
                    'origin_x_slider': {},
                    'origin_y_slider': {},
                    'origin_z_slider': {},
                    'norm_slider': {}, 
                    'pitch_slider': {}, 
                    'yaw_slider': {}, 
                    'roll_slider': {}, 
                    'min_intensity_slider': {}, 
                    'max_intensity_slider': {},
                    'pixel_spacing_x_input': {},
                    'pixel_spacing_y_input': {},
                    'slice_thickness_input': {}}
    
    
    OPTIONS_INFO = {'label': ['Image Number   ', 
                              'Origin X       ', 'Origin Y       ', 'Origin Z       ',
                              'Norm Location  ', 'Pitch Angle    ', 'Yaw Angle      ', 
                              'Roll Angle     ', 'Min Intensity  ', 'Max Intensity  ', 
                              'Pixel Spacing X', 'Pixel Spacing Y', 'Slice Thickness'], 
                    'default_value': [1, 
                                      0, 0, 0,
                                      0, 0, 0, 
                                      0, 0, 1200, 
                                      1.0, 1.0, 1.0], 
                    'previous_value': [1, 
                                       0, 0, 0,
                                       0, 0, 0, 
                                       0, 0, 1200, 
                                       1.0, 1.0, 1.0],
                    'current_value': [1, 
                                      0, 0, 0,
                                      0, 0, 0, 
                                      0, 0, 1200, 
                                      1.0, 1.0, 1.0],
                    'difference_value': [0, 
                                         0, 0, 0,
                                         0, 0, 0, 
                                         0, 0, 0, 
                                         0, 0, 0],
                    'min_value': [1, 
                                  -400, -400, -400,
                                  -400, -180, -180, 
                                  -180, -1e6, -1e6 + 1, 
                                  0.10, 0.10, 0.10],
                    'max_value': [1, 
                                  400, 400, 400,
                                  400, 180, 180, 
                                  180, 1e6 - 1, 1e6, 
                                  25.0, 25.0, 25.0],
                    'step_value': [1, 
                                   1.0, 1.0, 1.0,
                                   1.0, 1.0, 1.0, 
                                   1.0, 1.0, 1.0, 
                                   0.1, 0.1, 0.1],
                    'step_fast_value': [1, 
                                        5.0, 5.0, 5.0,
                                        5.0, 5.0, 5.0, 
                                        5.0, 5.0, 5.0, 
                                        0.5, 0.5, 0.5],
                    'option_type': ['int',   
                                    'float', 'float', 'float',
                                    'float', 'float', 'float', 
                                    'float', 'float', 'float', 
                                    'float', 'float', 'float'],
                    'group_tag': ['img_index_slider_group', 
                                  'origin_x_slider_group', 'origin_y_slider_group', 'origin_z_slider_group',
                                  'norm_slider_group', 'pitch_slider_group', 'yaw_slider_group', 
                                  'roll_slider_group', 'min_intensity_slider_group', 'max_intensity_slider_group',
                                  'pixel_spacing_x_input_group', 'pixel_spacing_y_input_group', 'slice_thickness_input_group'],
                    'label_tag': ['img_index_slider_label', 
                                  'origin_x_slider_label', 'origin_y_slider_label', 'origin_z_slider_label', 
                                  'norm_slider_label', 'pitch_slider_label', 'yaw_slider_label', 
                                  'roll_slider_label', 'min_intensity_slider_label', 'max_intensity_slider_label',
                                  'pixel_spacing_x_input_label', 'pixel_spacing_y_input_label', 'slice_thickness_input_label'],
                    'slider_tag': ['img_index_slider', 
                                   'origin_x_slider', 'origin_y_slider', 'origin_z_slider',
                                   'norm_slider', 'pitch_slider', 'yaw_slider', 
                                   'roll_slider', 'min_intensity_slider', 'max_intensity_slider',
                                   'pixel_spacing_x_input', 'pixel_spacing_y_input', 'slice_thickness_input']}
    print(f'GLOBALS Message:')
    for option_index, option_key in enumerate(OPTIONS_DICT.keys()):
        for info_key in OPTIONS_INFO.keys():
            OPTIONS_DICT[option_key][info_key] = OPTIONS_INFO[info_key][option_index]
            print(f'\t{option_key:<21}, {info_key:<16}: {OPTIONS_DICT[option_key][info_key]}')
            
    ORIENTATION_OPTIONS = ['origin_x_slider',
                           'origin_y_slider',
                           'origin_z_slider',
                           'norm_slider', 
                           'pitch_slider', 
                           'yaw_slider', 
                           'roll_slider',
                           'pixel_spacing_x_input',
                           'pixel_spacing_y_input',
                           'slice_thickness_input']
                           
    OPTION_TAGS = ['img_index_slider', 
                   'origin_x_slider',
                   'origin_y_slider',
                   'origin_z_slider',
                   'norm_slider', 
                   'pitch_slider', 
                   'yaw_slider', 
                   'roll_slider', 
                   'min_intensity_slider', 
                   'max_intensity_slider', 
                   'orientation_group_layer_control_button', 
                   'orientation_group_layer_reset_button',
                   'intensity_group_layer_control_button', 
                   'colormap_combo', 
                   'colormap_scale_combo',
                   'mask_combo',
                   'enable_mask_checkbox',
                   'mask_opacity_slider',
                   'mask_color_picker',
                   'reverse_colormap_checkbox',
                   'interpolation_combo_box',
                   'mask_exclude_low_float',
                   'enable_mask_low_exclude_checkbox',
                   'mask_exclude_high_float',
                   'enable_mask_high_exclude_checkbox',
                   'pixel_spacing_x_input',
                   'pixel_spacing_y_input',
                   'slice_thickness_input']
    
    IMAGE_TOOL_TAGS = ['add_drag_point_button']
    
    MASK_DICT = {'Air': 0,
                 'Body': 1,
                 'Bones': 2,
                 'Spine': 3,
                 'Ribs': 4,
                 'Lungs, Vasculature, Trachea': 5,
                 'Left Vasculature': 6,
                 'Right Vasculature': 7, 
                 'Lungs, Trachea': 8,
                 'Trachea': 9, 
                 'Lungs': 10,
                 'Left Lung': 11,
                 'Right Lung': 12}
    
    DEBUG_INFO_TAGS = ['current_group_info',
                       'current_volume_info',
                       'current_volume_name',
                       'current_group_name',
                       'current_volume_index',
                       'initial_origin_vector',
                       'initial_norm_vector',
                       'initial_quaternion',
                       'current_origin_vector',
                       'current_norm_vector',
                       'current_quaternion',
                       'current_view_slice']
    

    def add_configuration_to_value_registry(value_registry_tag):
        print('GLOBALS Message: Adding configuration values to value registry.')
        for section in G.CONFIG_DICT:
            for config_key, config_value in G.CONFIG_DICT[section].items():
                print(f'\t\tAdding {config_key} to value registry.')
                registry_kwargs = {'tag': f'ValueRegister_Configuration_{config_key}_value',
                                   'default_value': config_value,
                                   'parent': value_registry_tag}
                match config_value:
                    case int():
                        dpg.add_int_value(**registry_kwargs)
                    case float():
                        dpg.add_float_value(**registry_kwargs)
                    case str():
                        dpg.add_string_value(**registry_kwargs)
                    case tuple():
                        dpg.add_color_value(**registry_kwargs)

    def add_colormap_combo_to_value_registry(value_registry_tag):
        dpg.add_string_value(tag = 'colormap_combo_current_value',
                             default_value = 'Fire', 
                             parent = value_registry_tag)
        dpg.add_string_value(tag = 'colormap_scale_type_current_value',
                             default_value = 'Linear', 
                             parent = value_registry_tag)
        dpg.add_bool_value(tag = 'colormap_rescaled_current_value',
                           default_value = False,
                           parent = value_registry_tag)
        dpg.add_bool_value(tag = 'colormap_reversed_current_value',
                           default_value = False,
                           parent = value_registry_tag)

    def add_input_options_to_value_registry(value_registry_tag):
        # Set up slider value registry. This is used with the mouse wheel which can otherwise extend past the 
        # slider limits. 
        print('GLOBALS Message: Adding slider tags values to value registry.')
        for option_key in G.OPTIONS_DICT.keys():
            slider_tag = G.OPTIONS_DICT[option_key]['slider_tag']
            slider_value = G.OPTIONS_DICT[option_key]['default_value']
            step_value = G.OPTIONS_DICT[option_key]['step_value']
            step_fast_value = G.OPTIONS_DICT[option_key]['step_fast_value']
            option_type = G.OPTIONS_DICT[option_key]['option_type']
            print(f'\t\tAdding {slider_tag}_current_value to value registry.')
            print(f'\t\tAdding {slider_tag}_increment_input_float_value to value registry.')
           
            if option_type == 'int':
                dpg.add_int_value(tag = f'{slider_tag}_current_value', 
                                  default_value = 1*slider_value, 
                                  parent = value_registry_tag)
                
            elif option_type == 'float':
                dpg.add_float_value(tag = f'{slider_tag}_current_value', 
                                    default_value = 1.0*slider_value, 
                                    parent = value_registry_tag)
                dpg.add_float_value(tag = f'{slider_tag}_step_value', 
                                    default_value = 1.0*step_value, 
                                    parent = value_registry_tag)
                dpg.add_float_value(tag = f'{slider_tag}_step_fast_value', 
                                    default_value = 1.0*step_fast_value, 
                                    parent = value_registry_tag)
                dpg.add_float_value(tag = f'{slider_tag}_increment_input_float_value', 
                                    default_value = 1.0*step_value, 
                                    parent = value_registry_tag)
    
    def add_texture_center_to_value_registry(value_registry_tag,
                                             texture_center_value:int = TEXTURE_CENTER):

        print('GLOBALS Message: Adding texture center value to registry.')

        texture_center_tag = f'global_texture_center'

        if dpg.does_item_exist(texture_center_tag):
            dpg.delete_item(texture_center_tag)
            
        if dpg.does_alias_exist(texture_center_tag):
            dpg.remove_alias(texture_center_tag)

        dpg.add_int_value(tag = texture_center_tag,
                          default_value = texture_center_value,
                          parent = value_registry_tag)
        
    def set_texture_center_value(value):
        dpg.set_value('texture_center_tag', 
                      value)

    # def add_histogram_values_to_value_registry(value_registry_tag):
    #     dpg.add_float_vect_value(tag = 'InfoBoxTab_histogram_volume_values',
    #                              default = [0,0,0,0],
    #                              parent = value_registry_tag)