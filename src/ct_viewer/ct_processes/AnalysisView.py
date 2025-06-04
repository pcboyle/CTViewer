from .Globals import *

class AnalysisView:
    def __init__(self):

        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()

        self.plot_x_axis = 'analysis_view_plot_x_axis'
        self.plot_y_axis = 'analysis_view_plot_y_axis'
        self.colormap_scale = 'analysis_view_colormap_scale'
        self.mouse_hover_handler = 'analysis_view_mouse_hover_handler'
        self.information_box = 'analysis_view_information_box'
        self.mouse_position_text = 'analysis_view_mouse_position_text'


        with dpg.group(horizontal = True, horizontal_spacing = 0):
            dpg.add_child_window(width = G.CONFIG_DICT['app_settings']['main_pane_width'] - 65, 
                                 height =G.CONFIG_DICT['app_settings']['main_pane_height'],
                                 tag='AnalysisView_texture_window')
            dpg.add_drawlist(width = G.CONFIG_DICT['app_settings']['main_pane_width'] - 65, 
                             height =G.CONFIG_DICT['app_settings']['main_pane_height'],
                             tag = create_tag('AnalysisView', 'DrawList', 'Texture'),
                             parent = 'AnalysisView_texture_window')
            
            dpg.add_child_window(width = 65, 
                                 height = G.CONFIG_DICT['app_settings']['main_pane_height'],
                                 tag='AnalysisView_colorbar_window')
            dpg.add_drawlist(width = 65, 
                             height = G.CONFIG_DICT['app_settings']['main_pane_height'],
                             tag = create_tag('AnalysisView', 'DrawList', 'Colorbar'),
                             parent='AnalysisView_colorbar_window')
            
    def load_image(self, volume_name):
        # Want to add the texture to this window. 
        pass