from .Globals import *

class MenuBar:
    def __init__(self, value_registry = None):
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()

        self.VolumeLayerGroups = None
        self.OptionsPanel = None
        self.ImageTools = None
        self.InformationBox = None
        self.FileDialog = None
        self.NodeEditor = None
        self.classes_assigned = False

        with dpg.menu_bar(tag = 'MenuBar'):
            with dpg.menu(label = 'File', tag = 'MenuBarFile'):
                dpg.add_menu_item(label = 'Open Volumes', 
                                  tag = 'MenuBarFile_open', 
                                  callback = self.open_files)
                dpg.add_menu_item(label = 'Add Volumes', 
                                  tag = 'MenuBarFile_add_volume', 
                                  callback = self.open_files)
                dpg.add_menu_item(label = 'Close All', 
                                  tag = 'MenuBarFile_close', 
                                  callback = self.close_all)
                dpg.add_menu_item(label = 'Exit', 
                                  tag = 'MenuBarFile_exit', 
                                  callback = self.exit_app)
                
            with dpg.menu(label = 'Layers', tag = 'MenuBarLayers'):
                dpg.add_menu_item(label = 'New Group',
                                  tag = 'MenuBarLayers_create_new_group', 
                                  callback = self.create_new_group)
                
            with dpg.menu(label = 'Landmarks', tag = 'MenuBarLandmarks'):
                dpg.add_menu_item(label = 'Save Landmarks',
                                  tag = 'MenuBarLandmarks_save_landmarks', 
                                  callback = self.save_landmarks)
                dpg.add_menu_item(label = 'Load Landmarks',
                                  tag = 'MenuBarLandmarks_load_landmarks', 
                                  callback = self.load_landmarks)
            
            with dpg.menu(label = 'Analysis', tag = 'MenuBarAnalysis'):
                dpg.add_menu_item(label = 'Open Analysis Window', 
                                  tag = 'MenuBarAnalysis_open_analysis_window', 
                                  callback = self.open_analysis_window)
                dpg.add_menu_item(label = 'Open Node Editor', 
                                  tag = create_tag('MenuBar', 'MenuItem', 'NodeEditor'), 
                                  callback = self.open_node_editor)

            with dpg.menu(label = 'Settings', tag = 'MenuBarSettings'):
                dpg.add_menu_item(label='Configuration', 
                                  tag = 'MenuBarSettings_open_config_menu', 
                                  user_data = False,
                                  callback = self.open_configuration)
                dpg.add_menu_item(label='Help',
                                  tag = 'MenuBarSettings_open_help_dialog',
                                  user_data = False, 
                                  callback = self.open_help)
                with dpg.menu(label='Debug Options', 
                              tag = 'MenuBarSettings_open_debug_menu'):
                    dpg.add_menu_item(label = 'Show Debug', 
                                      tag = 'MenuBarDebugOptions_show_debug', 
                                      user_data = False, 
                                      callback = self.show_debug)
                    dpg.add_menu_item(label = 'Show Metrics', 
                                      tag = 'MenuBarDebugOptions_show_metrics',
                                      user_data = False, 
                                      callback = self.show_metrics)
                    dpg.add_menu_item(label = 'Show Item Registry', 
                                      tag = 'MenuBarDebugOptions_show_item_registry', 
                                      user_data = False, 
                                      callback = self.show_item_registry)
                    dpg.add_menu_item(label = 'Show Texture Registry', 
                                      tag = 'MenuBarDebugOptions_show_texture_registry', 
                                      user_data = False, 
                                      callback = self.show_texture_registry)
                    dpg.add_menu_item(label = 'Show About', 
                                      tag = 'MenuBarDebugOptions_show_about', 
                                      user_data = False, 
                                      callback = self.show_about)
                    dpg.add_menu_item(label = 'Show Documentation', 
                                      tag = 'MenuBarDebugOptions_show_documentation', 
                                      user_data = False, 
                                      callback = self.show_documentation)
                    dpg.add_menu_item(label = 'Show Value Registry',
                                      tag = 'MenuBarDebugOptions_show_value_registry',
                                      user_data = {'registry_hidden': False, 
                                                   'registry_tag': value_registry},
                                      callback = self.open_value_registry)


    def open_node_editor(self):
        self.NodeEditor.open()

    def show_about(self):
        dpg.show_about()

    def show_documentation(self):
        dpg.show_documentation()

    def show_metrics(self, sender, app_data, user_data):
        if G.SHOW_METRICS:
            return
        dpg.show_metrics()

    def show_debug(self, sender, app_data, user_data):
        if G.SHOW_DEBUG:
            return
        dpg.show_debug()

    def show_item_registry(self, sender, app_data, user_data):
        if G.SHOW_ITEM_REGISTRY:
            return
        dpg.show_item_registry()

    def show_texture_registry(self, sender, app_data, user_data):
        if user_data:
            return
        dpg.show_item(G.TEX_REG_TAG)

    def open_files(self, sender, app_data):
        self.FileDialog.show()

    def set_classes(self, 
                    VolumeLayerGroups,
                    InformationBox,
                    OptionsPanel,
                    FileDialog,
                    ImageTools,
                    NodeEditor) -> None:
        self.VolumeLayerGroups = VolumeLayerGroups
        self.InformationBox = InformationBox
        self.OptionsPanel = OptionsPanel
        self.FileDialog = FileDialog
        self.ImageTools = ImageTools
        self.NodeEditor = NodeEditor
        self.classes_assigned = True

    def close_all(self):
        print('MenuBar Message: Closing All Volumes')
        print(f'MenuBar Message: {self.VolumeLayerGroups.active = }')
        if self.VolumeLayerGroups.active:
            dpg.set_value('InfoBoxTab_layers_text', '')
            
            self.InformationBox.close_image()
            self.OptionsPanel.close_image()
            self.ImageTools.disable_options()
            # dpg.configure_item('save_landmarks_button', enabled = False)

            self.VolumeLayerGroups.remove_all_groups()

            setattr(self, 'VOLUME_LOADED', False)
            for GLOBAL_KEY in G.GLOBAL_DEFAULTS.keys():
                setattr(G, GLOBAL_KEY, G.GLOBAL_DEFAULTS[GLOBAL_KEY])
        
            print(f'MenuBar Message: {G.FILE_LOADED}, False')
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            print(f'MenuBar Message: {G.FILE_LOADED = }')
    

    def exit_app(self):
        print('MenuBar Message: Exiting App')
        dpg.stop_dearpygui()


    def create_new_group(self):
        pass
    

    def save_landmarks(self):
        if self.classes_assigned:
            self.VolumeLayerGroups.save_landmarks()


    def load_landmarks(self):
        if self.classes_assigned:
            print('MenuBar Message: Loading Landmarks.')
            loaded_landmarks = self.VolumeLayerGroups.load_landmarks()
            
            print('MenuBar Message: Adding loaded_landmarks.')
            for landmark_info in loaded_landmarks: 
                print(f'\t{landmark_info = }')
                self.InformationBox.add_landmark(*landmark_info)

            self.OptionsPanel.update_volume('load_landmarks', None, None)

    def open_analysis_window(self):
        if G.FILE_LOADED:
            pass
            
        else:
            pass
        

    def open_value_registry(self, sender, app_data, user_data):

        if user_data['registry_hidden']:
            dpg.show_item('MenuBar_ValueRegistryWindow')
            return
        
        else:
            user_data['registry_hidden'] = True

        value_label_tags = []

        with dpg.window(label='CTViewer Value Registry', 
                        tag = 'MenuBar_ValueRegistryWindow',
                        pos = [500, 100],
                        max_size=[950, 800],
                        horizontal_scrollbar = True,
                        autosize = True,
                        show = True):
            
            value_tag_label_length = 0

            for dpg_index in dpg.get_item_children(user_data['registry_tag'], slot=1):
                value_tag_alias = dpg.get_item_alias(dpg_index)
                value_tag_label_length = max(value_tag_label_length, len(f'{value_tag_alias}'))

                with dpg.group(horizontal = True):
                    dpg.add_text(value_tag_alias)
                    value_label_tags.append(dpg.last_item())
                    if isinstance(dpg.get_value(dpg_index), dict):
                        dpg.add_text('DICTIONARY_PLACEHOLDER')
                    else:
                        dpg.add_text(f'\t{dpg.get_value(dpg_index)}')
            for value_label_tag in value_label_tags:
                dpg.set_value(value_label_tag, f'{dpg.get_value(value_label_tag):<{value_tag_label_length}} :')

    def open_configuration(self, sender, app_data, user_data):

        if user_data:
            dpg.show_item('MenuBar_Configuration_Window')
            return
        
        else:
            dpg.set_item_user_data(sender, True)

        with dpg.window(label='CTViewer Configuration', 
                        tag = 'MenuBar_Configuration_Window',
                        pos = [500, 300],
                        autosize = True,
                        show = True):
            
            config_section_label_length = 0
            config_value_label_length = 0

            for section in G.CONFIG_DICT.keys():
                config_section_label_length = max(config_section_label_length, len(f'{section}'))
                with dpg.group():
                    dpg.add_text(f'{section}', tag = f'Configuration_{section}_group_label')
                    for config_key, config_value in G.CONFIG_DICT[section].items():
                        with dpg.group(horizontal=True):
                            dpg.add_text(default_value = f'{config_key}', tag = f'Configuration_{config_key}_value_label')
                            config_value_label_length = max(config_value_label_length, len(f'{config_key}'))
                            match config_value:
                                case int():
                                    dpg.add_input_int(tag = f'Configuration_{config_key}_value_input', 
                                                      default_value = config_value, 
                                                      user_data = [section, config_key],
                                                      min_value = 15, 
                                                      max_value = 10000, 
                                                      source = f'ValueRegister_Configuration_{config_key}_value',
                                                      callback=self.update_config)
                                case float():
                                    dpg.add_input_float(tag = f'Configuration_{config_key}_value_input', 
                                                        default_value = config_value, 
                                                        user_data = [section, config_key],
                                                        step = 1.0,
                                                        step_fast = 5.0,
                                                        min_value = 15.0, 
                                                        max_value = 10000.0, 
                                                        source = f'ValueRegister_Configuration_{config_key}_value',
                                                        callback=self.update_config)
                                case str():
                                    dpg.add_input_text(tag = f'Configuration_{config_key}_value_input', 
                                                       default_value = config_value, 
                                                       user_data = [section, config_key],
                                                       source = f'ValueRegister_Configuration_{config_key}_value',
                                                       callback=self.update_config)
                                    
                                case tuple():
                                    dpg.add_color_edit(tag = f'Configuration_{config_key}_value_input', 
                                                       default_value = config_value, 
                                                       user_data = [section, config_key],
                                                       source = f'ValueRegister_Configuration_{config_key}_value',
                                                       callback = self.update_config)

            # Set lengths of labels
            for section in G.CONFIG_DICT.keys():
                dpg.set_value(f'Configuration_{section}_group_label', f'{section:<{config_section_label_length}}:')
                for config_key in G.CONFIG_DICT[section].keys():
                    dpg.set_value(f'Configuration_{config_key}_value_label', f'{config_key:<{config_value_label_length}}:')

            dpg.add_button(label = 'Save Configuration', callback = self.save_configuration)


    def open_help(self, sender, app_data, user_data):
        if user_data:
            dpg.show_item('MenuBar_HelpWindow')
            return

        else: 
            dpg.set_item_user_data(sender, True)
        
        with dpg.window(label='CTViewer Help', 
                        tag = 'MenuBar_HelpWindow',
                        pos = [500, 100],
                        max_size=[950, 800],
                        horizontal_scrollbar = False,
                        autosize = True,
                        show = True):
            dpg.add_text(self.get_menubar_text(),
                         wrap = 795)
            
            dpg.set_item_user_data(sender, True)


    def update_config(self, sender, app_data, user_data):
        if user_data[0] == 'default_colors':
            app_data = tuple([int(round(value*255.0)) for value in app_data])
        print(f'Config Message: Received {sender}. \n\tChanging {user_data} to {app_data}')
        G.CONFIG_DICT[user_data[0]][user_data[1]] = app_data


    def save_configuration(self, sender):
        G.save_config('current')

    def _cleanup_(self):
        pass

    def get_menubar_text(self):

        return """Command line options: 
            To start program, use ct_viewer. 

            Can enter debug mode with -debug

            Select GPU with -gpu=<device number>
                Default is -gpu=0

        Config file:

            A configuration file will now be created/loaded at $USERHOME/.config/ct_viewer/config.ini


        Geometry:
            All files are loaded in and interpolated to be 1 x 1 x 1 mm/pix using the pixel dimensions, slice thickness, and volume affine. They are initially centered at (0, 0, 0) in physical space in mm. 

            If no affine is present, assumes a 1x1x1 mm^3 voxel dimension and positively aligned volume. 

        Navigation: 
            All navigation is done from the perspective of the patient according to the affine of the volume. 
            
                    a       -   Move volume in the negative X direction   -   Patient moves to their Right
                    d       -   Move volume in the positive X direction   -   Patient moves to their Left

                    w       -   Move volume in the negative Y direction   -   Patient moves to their Anterior
                    s       -   Move volume in the positive Y direction   -   Patient moves to their Posterior

                    q       -   Move volume in the negative Z direction   -   Patient moves to their Inferior
                    e       -   Move volume in the positive Z direction   -   Patient moves to their Superior

                    z       -   Negative change in image index            -
                    c       -   Positive change in image index            -

            Shift + a       -   Negative roll of volume                   -   Rotates about Z, moves Y into X. 
            Shift + d       -   Positive roll of volume                   -   Rotates about Z, moves X into Y. 

            Shift + w       -   Negative pitch of volume                  -   Rotates about X, moves Z into Y. 
            Shift + s       -   Positive pitch of volume                  -   Rotates about X, moves Y into Z. 

            Shift + q       -   Negative yaw of volume                    -   Rotates about Y, moves Z into X. 
            Shift + e       -   Positive yaw of volume                    -   Rotates about Y, moves X into Z. 

            Shift + z       -   Swaps between Group and Local control     -  
            Shift + c       -   Swaps between Group and Local control     - 

            Alt + a         -   Negative zoom in X direction              -   
            Alt + d         -   Positive zoom in X direction              -   

            Alt + w         -   Negative zoom in Y direction              -   
            Alt + s         -   Positive zoom in Y direction              -   
            
            Alt + q         -   Negative zoom in Z direction              -   
            Alt + e         -   Positive zoom in Z direction              -   

            Crtl + Any      -   Changes navigation setting at 5x the current increment. 
                            -   Does not effect z and c keys. 

            Middle Mouse    -   Click and drag to move the volume. Currently a little floaty as the rendering and mouse motion aren't perfectly aligned. 

            Spacebar        -   Create landmark at crosshair position"""