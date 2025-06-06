class CTViewer:
    # Main window chosen in ct_viewer.py, which contains the main() class. 
    def __init__(self, main_window):

        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()

        from . import MenuBar
        from . import CTVolume
        from . import OptionsPanel
        from . import NodeEditor
        from . import NewMainView
        from . import AnalysisView
        from . import FileDialog
        from . import InformationBox
        from . import VolumeLayer
        from . import ImageTools
        from . import Themes

        # Set this instance as the ct_viewer. 
        print('Initialized')
        G.APP = self

        self.VolumeLayerGroups:VolumeLayer.VolumeLayerGroups = VolumeLayer.VolumeLayerGroups()
        self.VolumeLayerGroups.add_group(group_name = 'AllVolumes')
        self.main_view:NewMainView.MainView = NewMainView.MainView(self.VolumeLayerGroups)
        self.texture_registry = dpg.add_texture_registry(tag = G.TEX_REG_TAG) # 'main_texture_registry'
        self.colormap_registry = dpg.add_colormap_registry(tag = G.COLORMAP_TAG)
        self.value_registry = dpg.add_value_registry(tag = G.VALUE_REG_TAG)
        self.handler_registry = dpg.add_handler_registry(tag = G.HANDLER_REG_TAG)
        # Need an item handler registry for hovering only. 
        self.item_handler_registry = dpg.add_item_handler_registry(tag = G.ITEM_HANDLER_REG_TAG)
        self.item_hovered_registry = dpg.add_item_handler_registry(tag = 'item_hovered_registry')
        self.themes = Themes.Themes()
        self.themes.register_colormaps(G.COLORMAP_DICT,
                                       self.colormap_registry)
        detect_drives = False
        if f'{sys.platform}' == 'win32':
            detect_drives = True
        self.FileDialog = FileDialog.FileDialog(G.CONFIG_DICT['directories']['image_dir'], 
                                                has_parent = True,
                                                debug = True, 
                                                show = False,
                                                detect_drives = detect_drives)
        
        self.NodeEditor = NodeEditor.NodeEditor()

        G.add_input_options_to_value_registry(self.value_registry)
        G.add_configuration_to_value_registry(self.value_registry)
        G.add_texture_center_to_value_registry(self.value_registry)
        
        # Load up interpolators.
        CTVolume._initialize_interps()
        
        with main_window:
            self.menu_bar = MenuBar.MenuBar(value_registry = self.value_registry)
            with dpg.group(tag = 'RootWindowGroup',
                           horizontal = True):
                with dpg.child_window(tag = 'Navigation_Window',
                                      width = G.CONFIG_DICT['app_settings']['navigation_window_width'], 
                                      height = G.CONFIG_DICT['app_settings']['navigation_window_height'],
                                      border = True, 
                                      no_scrollbar = True):
                    with dpg.tab_bar(tag = G.LEFT_TAB_BAR, 
                                     callback = self.print_current_tab):
                        with dpg.tab(label = 'Navigation', 
                                     tag = G.NAVIGATION_TAB_TAG):
                            with dpg.group(tag = 'OptionsPanel_ImageTools_Group'):
                                self.options_panel = OptionsPanel.OptionsPanel(debug = True)
                                self.image_tools = ImageTools.ImageTools(self.VolumeLayerGroups)
                        # with dpg.tab(label = 'File Explorer', tag = G.FILE_EXPLORER_TAB_TAG):
                        #     self.file_explorer = FileExplorer.FileExplorer()
                with dpg.child_window(tag = 'MainViewTexture_Window',
                                      width = G.CONFIG_DICT['app_settings']['tab_pane_width'], #G.MAIN_TAB_VIEW_WINDOW_DEFAULTS['WINDOW_WIDTH'],
                                      height = G.CONFIG_DICT['app_settings']['tab_pane_height'], #G.MAIN_TAB_VIEW_WINDOW_DEFAULTS['WINDOW_HEIGHT'],
                                      border = True):
                    with dpg.tab_bar(tag = G.TAB_BAR_TAG, 
                                     callback = self.print_current_tab):
                        with dpg.tab(label = 'Volume Tab', 
                                     tag = G.VOLUME_TAB_TAG):
                            self.main_view.create_draw_window('MainTexture', 
                                                              G.VOLUME_TAB_TAG, 
                                                              item_handler_reg_tag=self.item_hovered_registry)
                        with dpg.tab(label = 'Analysis Tab', 
                                     tag = G.ANALYSIS_TAB_TAG):
                            self.analysis_view = AnalysisView.AnalysisView()
                
                self.info_box = InformationBox.InformationBox(self.VolumeLayerGroups)

        self.image_tools.set_options_panel(self.options_panel)
        self.options_panel.set_volume_and_draw_objects(self.VolumeLayerGroups,
                                                       self.main_view)
        
        self.options_panel.set_info_box(self.info_box)
        self.options_panel.set_image_tools(self.image_tools)
        
        self.FileDialog.initialize(volume_layer_groups = self.VolumeLayerGroups,
                                   draw_window = self.main_view,
                                   options_panel = self.options_panel,
                                   information_box = self.info_box,
                                   debug = G.DEBUG_MODE)
        
        dpg.add_item_hover_handler(parent = self.item_hovered_registry, 
                                   user_data = self.main_view.return_mouse_pos_texture_info_text_tag(),
                                   callback = self.VolumeLayerGroups.update_mouse_volume_coord_info)
        dpg.bind_item_handler_registry(self.main_view.return_texture_drawlist_tag(self.main_view.window_tag),
                                       self.item_hovered_registry)
        
        dpg.add_key_press_handler(key = dpg.mvKey_None, 
                                  callback = self.options_panel.mouse_and_keyboard_navigation, 
                                  tag = 'mouse_and_keyboard_navigation_handler', 
                                  parent = self.handler_registry)
        
        self.menu_bar.set_classes(self, 
                                  self.VolumeLayerGroups,
                                  self.info_box,
                                  self.options_panel,
                                  self.FileDialog,
                                  self.image_tools)

    def print_current_tab(self, sender, app_data):
        print('Tab clicked!')
        print('Sender: ', sender)
        print('App Data: ', app_data)

    def _cleanup_(self):
        dict_keys = list(self.__dict__.keys())
        while len(dict_keys) > 0:
            attrib_key = dict_keys.pop()
            try: 
                getattr(self, attrib_key)._cleanup_()
            except:
                pass
            finally:
                setattr(self, attrib_key, None)
                delattr(self, attrib_key)
        
from .Globals import *