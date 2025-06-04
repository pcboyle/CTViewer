from .Globals import *

class NodeEditor(object):
    
    def __init__(self):
        self.tag = create_tag('NodeEditor', 'Window', 'NodeEditor')
        with dpg.window(label = 'Node Editor',
                        tag = self.tag, 
                        width = -1, 
                        height = 750, 
                        show = False, 
                        menubar = False, 
                        modal = False):

            with dpg.node_editor(callback=self.link_callback, delink_callback=self.delink_callback):
                with dpg.node(label="Operation", tag = create_tag('NodeEditor', 'Node', 'Operation')):
                    
                    with dpg.node_attribute(label="Node A1"):
                        dpg.add_combo(default_value='Add', items = ['Add', 'Subtract', 'Divide', 'Multiply', 'Register'])

                    with dpg.node_attribute(label = 'Image 1', tag = create_tag('NodeEditor', 'NodeAttribute', 'Image1')):
                        dpg.add_combo(default_value='', items = [''])

                    with dpg.node_attribute(label = 'Image 2', tag = create_tag('NodeEditor', 'NodeAttribute', 'Image2')):
                        dpg.add_combo(default_value='', items = [''])

                with dpg.node(label = "Morphological", tag = create_tag('NodeEditor', 'Node', 'Morphological')):
                    with dpg.node_attribute(label='Node A2'):
                        dpg.add_combo(label='Image', tag = create_tag('NodeEditor', 'NodeAttribute', 'MorphologicalCombo'),
                                      default_value = '', items = ['Dilation', 'Erosion'])

                with dpg.node(label="Node 2"):
                    with dpg.node_attribute(label="Node A3"):
                        dpg.add_input_float(label="F3", width=200)

                    with dpg.node_attribute(label="Node A4", attribute_type=dpg.mvNode_Attr_Output):
                        dpg.add_input_float(label="F4", width=200)

    # callback runs when user attempts to connect attributes
    def link_callback(self, sender, app_data):
        # app_data -> (link_id1, link_id2)
        dpg.add_node_link(app_data[0], app_data[1], parent=sender)

    # callback runs when user attempts to disconnect attributes
    def delink_callback(self, sender, app_data):
        # app_data -> link_id
        dpg.delete_item(app_data)

    def open_node_editor(self):
        dpg.show_item(self.tag)

    pass