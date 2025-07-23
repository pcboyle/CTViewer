from .Globals import *
from . import CTVolume
from . import DICOM_Tags
from . import VolumeLayer
from . import NewMainView
from . import OptionsPanel
from . import InformationBox

INDENT_STEP = 14    # actually depends on font size

#TODO: Parse folders and directories using asynchronous workers to speed up process. 

def on_row_clicked(sender, value, user_data, selected_leaves:list = None):
    # Make sure it happens quickly and without flickering
    with dpg.mutex():
        table, row, leaf = user_data
        root_level, node, node_label = dpg.get_item_user_data(row)
        # Get selected data and add it to the table's user data. 
        if leaf:

            if type(selected_leaves) == type(None):
                current_indent, selected_leaves = dpg.get_item_user_data(table)

            else:
                selected_leaves = []
                current_indent = 0

            if value:
                selected_leaves.append(row)
            else:
                try:
                    selected_leaves.remove(row)
                except:
                    pass
            
            dpg.set_item_user_data(table, [current_indent, selected_leaves])

            print('FileDialog Message: Selected leaves:')
            for row_leaf in selected_leaves:
                print(f'FileDialog Message: \t{row_leaf}, {dpg.get_item_user_data(dpg.get_item_user_data(row_leaf)[1])[0]}')

            return

        # We don't want to highlight the selectable as "selected"
        dpg.set_value(sender, False)
        root_level, node, node_label = dpg.get_item_user_data(row)

        # First of all let's toggle the node's "expanded" status
        is_expanded = not dpg.get_value(node)
        dpg.set_value(node, is_expanded)
        # All children *beyond* this level (but not on this level) will be hidden
        hide_level = 10000 if is_expanded else root_level

        # Now manage the visibility of all the children as necessary
        rows = dpg.get_item_children(table, slot=1)
        root_idx = rows.index(row)
        # We don't want to look at rows preceding our current "root" node
        rows = rows[root_idx + 1:]
        for child_row in rows:
            child_level, child_node, child_label = dpg.get_item_user_data(child_row)
            if child_level <= root_level:
                break
            if child_level > hide_level:
                dpg.hide_item(child_row)
            else:
                dpg.show_item(child_row)
                hide_level = 10000 if dpg.get_value(child_node) else child_level


def delete_node(parent_table, parent_row):
    with dpg.mutex():
        root_level, node, node_label = dpg.get_item_user_data(parent_row)
        print(f'FileDialog Message: Deleting {parent_row}')

        rows = dpg.get_item_children(parent_table, slot = 1)
        root_idx = rows.index(parent_row)

        rows = rows[root_idx + 1:]
        for child_row in rows:
            child_level, child_node, child_label = dpg.get_item_user_data(child_row)
            if child_level <= root_level:
                break
            else:
                dpg.delete_item(child_row)
                current_indent, selected_leaves = dpg.get_item_user_data(parent_table)
                try:
                    selected_leaves.remove(child_row)
                except:
                    pass
                dpg.set_item_user_data(parent_table, [current_indent, selected_leaves])


        dpg.delete_item(parent_row)


def toggle_node(parent_table, parent_row):
    with dpg.mutex():
        root_level, node, node_label = dpg.get_item_user_data(parent_row)

        is_expanded = not dpg.get_value(node)
        dpg.set_value(node, is_expanded)

        hide_level = 10000 if is_expanded else root_level

        rows = dpg.get_item_children(parent_table, slot=1)
        root_idx = rows.index(parent_row)

        rows = rows[root_idx + 1:]
        for child_row in rows:
            child_level, child_node, child_label = dpg.get_item_user_data(child_row)
            if child_level <= root_level:
                break
            if child_level > hide_level:
                dpg.hide_item(child_row)
            else:
                dpg.show_item(child_row)
                hide_level = 10000 if dpg.get_value(child_node) else child_level

@contextmanager
def table_tree_node(*cells: str, leaf: bool = False, 
                    tag: int|str = None, parent_table: int|str = None, node_user_data = None,
                    add_popup: bool = False) -> Generator[Union[int, str], None, None]:
    if type(parent_table) == type(None):
        parent_table = dpg.top_container_stack()
    current_indent_level, current_selected_list = dpg.get_item_user_data(parent_table) or [0, []]
    node = dpg.generate_uuid() if type(tag) == type(None) else tag
    
    with dpg.table_row(user_data=(current_indent_level, node, cells[0]), parent = parent_table) as row:
        with dpg.group(horizontal=True, horizontal_spacing=0):
            dpg.add_selectable(span_columns=True, callback=on_row_clicked, user_data=(parent_table, row, leaf))
            if add_popup:
                with dpg.popup(dpg.last_item()):
                    dpg.add_text(f'Popup on row {row}.')
            dpg.add_tree_node(
                    tag=node,
                    label=cells[0],
                    indent=current_indent_level*INDENT_STEP,
                    leaf=leaf,
                    bullet=leaf,
                    default_open=True,
                    user_data = node_user_data
                    )

        for label in cells[1:]:
            dpg.add_text(label)

    try:
        dpg.set_item_user_data(parent_table, [current_indent_level + 1, current_selected_list])
        # print(f'FileDialog Message: {"\t"*(cur_level + 1)}Try    : User Data for {parent_table}-{cells[0]}: {cur_level + 1}')
        yield node, row, cells[0]

    finally:
        dpg.set_item_user_data(parent_table, [current_indent_level, current_selected_list])
        # print(f'FileDialog Message: {"\t"*cur_level}Finally: User Data for {parent_table}-{cells[0]}: {cur_level}')


def add_table_tree_leaf(*cells: str, parent_table: int|str = None, tag: int|str = None, node_user_data = None) -> Union[int, str]:
    with table_tree_node(*cells, leaf = True, parent_table = parent_table, tag = tag, node_user_data = node_user_data) as node:
        pass
    return node


def create_tree_node(node_dict:dict, parent_table: int|str = None, tag: int|str = None, 
                     row_list = [], add_popup = False, node_user_data:list = ['', dict()]):
    """
    Expects a dictionary of form: 

    {'Name': name,
     'Type': 'Node'|'Leaf', 'Group'|'Dataset',
     'Data': {'Data_key_1': {'Name': name,
                             'Type': 'Node'|'Leaf', 'Group'|'Dataset',
                             'Data': ...,
                             'Attributes': ...}}}
    """
     # t_node = (node, row, cells[0])
    if node_dict['Type'] in ['Node','Group']:
        if node_user_data[0] == '':
            node_user_data = [f'{node_dict['Name']}', node_dict['Attributes']]
        else:
            node_user_data = [f'{node_user_data[0]}|{node_dict['Name']}', node_dict['Attributes']]
        with table_tree_node(node_dict['Name'], '', '', parent_table = parent_table, add_popup = add_popup) as t_node:
            for node_data in node_dict['Data'].values():
                create_tree_node(node_data, parent_table = parent_table, row_list = row_list, add_popup = add_popup, node_user_data = node_user_data)
        toggle_node(parent_table, t_node[1])
        node_dict['Row'] = t_node[1]

    elif node_dict['Type'] in ['Leaf','Dataset']:
        node_user_data = [f'{node_user_data[0]}|{node_dict['Name']}', node_dict['Attributes']]
        t_node = add_table_tree_leaf(node_dict['Name'], 
                                     f'{node_dict['Attributes']['shape']}', 
                                     f'{node_dict['Attributes']['dtype']}',
                                     parent_table = parent_table,
                                     node_user_data = node_user_data)
        row_list.append(t_node[1])
        node_dict['Row'] = t_node[1]

    else:
        print(f'FileDialog Message: Unknown object type {node_dict["Type"]}')

    return row_list

class FileDialog(object):
    
    size_string_dict = {0.0: 'B',
                        1.0: 'B',
                        2.0: 'B',
                        3.0: 'KB',
                        4.0: 'KB',
                        5.0: 'KB',
                        6.0: 'MB',
                        7.0: 'MB',
                        8.0: 'MB',
                        9.0: 'GB',
                        10.0: 'GB',
                        11.0: 'GB',
                        12.0: 'TB',
                        13.0: 'TB',
                        14.0: 'TB'}

    file_type_dict = {'.txt': 'Text File',
                      '.py': 'Python Source File',
                      '.mat': 'Matlab File',
                      '.h5': 'HDF5 File', 
                      '.hdf5': 'HDF5 File',
                      '.nii': 'Nifti File',
                      '.nii.gz': 'gzipped Nifti File'}
    
    matfile_version_dict = {'(0, 0)': ['v4', 'mat'],
                            '(1, 0)': ['v5', 'mat'],
                            '(2, 0)': ['v7.3', 'hdf5']}
    
    file_info_formatter = {'File': lambda x: x.as_posix(),
                           'Parent': lambda x: x.as_posix(),
                           'Grandparent': lambda x: x.as_posix(),
                           'Relative Path': lambda x: x.as_posix(),
                           'Permissions': lambda x: oct(x)[2:],
                           'Size': lambda x: FileDialog.format_file_size(x), 
                           'Last Accessed': lambda x: f'{x[1]}',
                           'Last Modified': lambda x: f'{x[1]}',
                           'Creation Time': lambda x: f'{x[1]}'}
    
    file_info_object = {'File': lambda x: x,
                        'Parent': lambda x: x.parent,
                        'Grandparent': lambda x: x.parent.parent,
                        'Relative Path': lambda x: x.relative_to(x.parent.parent),
                        'Name': lambda x: x.name,
                        'Stem': lambda x: FileDialog.parse_file_suffix(x)[0],
                        'Suffix': lambda x: FileDialog.parse_file_suffix(x)[1],
                        'Is File': lambda x: x.is_file(),
                        'Is Dir': lambda x: x.is_dir(),
                        'Permissions': lambda x: x.stat().st_mode,
                        'File ID': lambda x: FileDialog.get_file_id(x),
                        'Size': lambda x: x.stat().st_size,
                        'Formatted Size': lambda x: FileDialog.format_file_size(x.stat().st_size),
                        'Last Accessed': lambda x: (x.stat().st_atime, datetime.datetime.fromtimestamp(np.round(x.stat().st_atime))),
                        'Last Modified': lambda x: (x.stat().st_mtime, datetime.datetime.fromtimestamp(np.round(x.stat().st_mtime))),
                        'Creation Time': lambda x: FileDialog.get_creation_time(x),
                        'File Type': lambda x: FileDialog.determine_file_type(x),
                        'Dicom Dir': lambda x: FileDialog.detect_dicom_dir(x),
                        'Is Mount': lambda x: x.is_mount(),
                        'Anchor': lambda x: x.anchor}
    
    file_info_list = ['Relative Path', 'Name', 'Suffix', 'Size', 'Formatted Size', 'Last Modified']

    column_names_to_index = {'File': 0,
                             'Parent': 1,
                             'Grandparent': 2,
                             'Relative Path': 3,
                             'Name': 4,
                             'Stem': 5,
                             'Suffix': 6,
                             'Is File': 7,
                             'Is Dir': 8,
                             'Permissions': 9,
                             'File ID': 10,
                             'Size': 11,
                             'Formatted Size': 11,
                             'Last Accessed': 13,
                             'Last Modified': 14,
                             'Creation Time': 15,
                             'File Type': 16,
                             'Dicom Dir': 17,
                             'Is Mount': 18,
                             'Anchor': 19}

    def detect_dicom_dir(file_path: Path):
        """
        Detects files in a DICOM directory. Can only handle image files that contain the following fields:
            seriesUID
            bit_depth
            rows
            cols
            direction_cosines
            affine
            slice_thickness
            pixel_spacing
        """
        with dpg.mutex():
            dcm_files = sorted(list(file_path.glob('*.dcm')))
            n_files = len(dcm_files)
            dcm_dir_contents = None
            # print(f'FileDialog Message: {file_path}\n\t{n_files}')
            is_dicom_dir = False
            if n_files > 0:
                is_dicom_dir = True
                dcm_dir_contents = {}
                print(f'FileDialog Message:\tDetect Dicom Dir {file_path.name}: {n_files}')
                gdcm_dir = gdcm.Directory()
                gdcm_dir.Load(f'{file_path}')
                file_names = sorted(gdcm_dir.GetFilenames())
                scanner = gdcm.Scanner.New()
                
                for keyword, comma_separated_tag in DICOM_Tags.KEYWORDS_TO_TAGS.items():
                    add_tag = gdcm.Tag()
                    add_tag.ReadFromCommaSeparatedString(comma_separated_tag)
                    scanner.AddTag(add_tag)

                scanned = scanner.Scan(file_names)

                im_pos_array = np.zeros((n_files,3), dtype = np.float32)
                slice_locations = np.zeros(n_files, dtype = np.float32)

                for file_index, file in enumerate(file_names):
                    pttv = gdcm.PythonTagToValue(scanner.GetMapping(file))
                    pttv.Start()

                    while (not pttv.IsAtEnd()):
                        tag:str = pttv.GetCurrentTag()
                        value:str = pttv.GetCurrentValue()
                        match tag.PrintAsContinuousString():
                            case "0020000e":
                                seriesUID = f'{value.strip()}'
                            case "00280100":
                                bit_depth = int(value.strip())
                            case "00280010":
                                rows = int(value.strip())
                            case "00280011":
                                cols = int(value.strip())
                            case "00200032":
                                stripped_value = value.strip().split('\\')
                                stripped_floats = [float(sv) for sv in stripped_value]
                                im_pos_array[file_index][:] = np.array(stripped_floats)[:]
                            case "00200037":
                                stripped_value = value.strip().split('\\')
                                stripped_floats = [float(sv) for sv in stripped_value]
                                dir_cos = np.array(stripped_floats)
                            case "00180050":
                                slice_thickness = float(value.strip())
                            case "00280030":
                                pix_spacing = np.array(value.strip().split('\\'), dtype = float)
                            case "00201041":
                                slice_locations[file_index] = float(value.strip())
                        pttv.Next()

                    if f'{seriesUID}' not in dcm_dir_contents:
                        dcm_dir_contents[f'{seriesUID}'] = {'files': [], 
                                                            'dir': file_path,
                                                            'shape': [cols, rows, 0], 
                                                            'dtype': f'int{bit_depth}',
                                                            'im_pos': np.zeros((n_files, 3)),
                                                            'slice_location': np.zeros(n_files),
                                                            'direction_cosines': np.zeros(6),
                                                            'affine': np.eye(4,4)}

                    dcm_dir_contents[seriesUID]['files'].append(file)
                    dcm_dir_contents[seriesUID]['shape'][2] += 1
                    dcm_dir_contents[seriesUID]['im_pos'][:] = im_pos_array[:] # X, Y, Z
                    dcm_dir_contents[seriesUID]['slice_location'][:] = slice_locations[:]

                dcm_dir_contents[seriesUID]['direction_cosines'][:] = dir_cos[:]
                dcm_dir_contents[seriesUID]['affine'][:3,0] = dir_cos[:3]*pix_spacing[1] # Delta Col
                dcm_dir_contents[seriesUID]['affine'][:3,1] = dir_cos[3:]*pix_spacing[0] # Delta Row
                dcm_dir_contents[seriesUID]['affine'][:3,2] = dcm_dir_contents[seriesUID]['im_pos'][1] - dcm_dir_contents[seriesUID]['im_pos'][0] + 0.0 # Slice Thickness
                dcm_dir_contents[seriesUID]['affine'][:3,3] = dcm_dir_contents[seriesUID]['im_pos'][0] + 0.0

                del gdcm_dir
                del scanner
            
            if is_dicom_dir:
                print(f'FileDialog Message: Dicom file {file_path.name} Affine:')
                for affine_element in dcm_dir_contents[seriesUID]['affine']:
                    print(f'\t{affine_element}')
                print(f'FileDialog Message: Image Position Row: {dcm_dir_contents[seriesUID]['im_pos'][0]}')
                print(f'FileDialog Message: Image Position Col: {dcm_dir_contents[seriesUID]['im_pos'][1]}')
                print(f'FileDialog Message: Delta X           : {dir_cos[:3]*pix_spacing[1]}')
                print(f'FileDialog Message: Delta Y           : {dir_cos[3:]*pix_spacing[0]}')

            return dcm_dir_contents


    def format_file_size(f_size):
        if float(f_size) == 0.0:
            f_size_power_of_ten = 0.0
        else:
            f_size_power_of_ten = np.floor(np.log10(float(f_size)))
        f_size_suffix = FileDialog.size_string_dict[f_size_power_of_ten]
        f_size_string = ''.join(f'{fs} ' for fs in [float(np.round(float(f_size) / np.power(10, f_size_power_of_ten), decimals = 3)), f_size_suffix])
        return f_size_string.rstrip()

    def format_file_info_string(file_info_dict):
        info_string = ''
        for key, value in file_info_dict.items():
            info_string = f'{info_string}\n{key:<16}: {value}'
        
        return info_string
    
    def parse_file_suffix(file_path:Path):
        """
        Returns
        """
        if file_path.is_file():
            for file_extension in FileDialog.file_type_dict.keys():
                if file_extension in file_path.name:
                    if file_extension == file_path.name[-len(file_extension):]:
                        return file_path.name.split(file_extension)[0], file_extension
            return file_path.stem, file_path.suffix
        else:
            return file_path.stem, ''
        
    def get_creation_time(file_path:Path):
        if file_path.stat().__contains__('st_birthtime'):
            return (file_path.stat().st_birthtime, datetime.datetime.fromtimestamp(np.round(file_path.stat().st_birthtime)))
        else:
            return (file_path.stat().st_ctime, datetime.datetime.fromtimestamp(np.round(file_path.stat().st_ctime)))

    def determine_file_type(file_path:Path):
        if file_path.is_file():
            suffix = FileDialog.parse_file_suffix(file_path)[1]
            match suffix:
                case '.mat':
                    return FileDialog.matfile_version_dict[f'{matfile_version(file_path)}'][1]
                case '.h5':
                    return 'hdf5'
                case '.hdf5':
                    return 'hdf5'
                case '.dcm':
                    return 'dicom'
                case '.nii':
                    return 'nifti'
                case '.nii.gz':
                    return 'nifti'
                case _:
                    return suffix
                
        elif file_path.is_dir():
            return 'dir'
        
        return ''
    
    def get_file_id(path: Path):
        return f'{path.stat().st_ino}-{path.stat().st_dev}'
    
    def get_windows_drive_letters():
        uppercase_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                             'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                             'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        drive_letters = []

        for letter in uppercase_letters:
            if Path(f"{letter}:\\").exists():
                drive_letters.append(Path(f'{letter}:\\').as_posix())
        
        return drive_letters
    
    def get_windows_drive_parent(drive_letters):
        pass

    def __init__(self, 
                 path: Path, 
                 has_parent:bool = False, 
                 debug: bool = True,
                 show: bool = True,
                 detect_drives = False):
        
        self.current_directory = Path(path)
        self.has_parent = has_parent
        self.debug = debug
        self.VolumeLayerGroups: VolumeLayer.VolumeLayerGroups = None
        self.InformationBox: InformationBox.InformationBox = None
        self.drawlist_tag = '',
        self.current_directory_file_dict = {}
        self.selected_files_dict = {}
        self.file_extension_dict = {"All Files (*)": [[''], []],
                                    "Image Files (.mat, .dcm, .h5, .hdf5, .nii, .nii.gz)": [['.mat', '.dcm', '.h5', '.hdf5', '.nii', '.nii.gz'], []],
                                    "MATLab Files (.mat)": [['.mat'], []],
                                    "DICOM Files (.dcm)": [['.dcm'], []],
                                    "HDF5 Files (.h5, .hdf5)": [['.h5', '.hdf5'], []],
                                    "Nifti Files (.nii, .nii.gz)": [['.nii', '.nii.gz'], []]}
        
        self.drive_letters = []
        if detect_drives: 
            self.drive_letters = FileDialog.get_windows_drive_letters()
            print('FileDialog Message: Drive Letters: ')
            print(f'\t{self.drive_letters}')

        self.parse_files_in_path(self.current_directory)
        
        self.table_row_index_to_tag = {} # Row index: row_tag
        self.table_row_tag_to_index = {} # Row_tag: row_index

        self.separator = '|'
        self.dpg_items = []
        self.file_table_columns = ['Name', 'Formatted Size', 'Last Modified', 'Suffix']
        self.file_dialog_table_tag = create_tag('FileDialog', 'Table', 'FileDialog')
        self.chosen_files_table_tag = create_tag('FileDialog', 'Table', 'ChosenFiles')
        self.item_handler_registry_tag = create_tag('FileDialog', 'ItemHandlerRegistry', 'TableRows')
        self.double_click_handler_tag = create_tag('FileDialog', 'DoubleClickHandler', 'TableRows')
        self.single_click_handler_tag = create_tag('FileDialog', 'SingleClickHandler', 'TableRows')
        self.value_registry_tag = create_tag('FileDialog', 'ValueRegistry', 'Strings')
        self.navigation_buttons_group_tag = create_tag('FileDialog', 'Group', 'NavigationButtons')
        self.current_directory_text_tag = create_tag('FileDialog', 'Text', 'CurrentDirectory')
        self.go_up_directory_button_tag = create_tag('FileDialog', 'Button', 'GoUpDirectory')
        self.refresh_directory_button_tag = create_tag('FileDialog', 'Button', 'RefreshDirectory')
        self.file_filter_combobox_tag = create_tag('FileDialog', 'ComboBox', 'FileExtensions')
        self.selected_rows_info_string = ''
        self.selected_rows_info_text_tag = create_tag('FileDialog', 'Text', 'SelectedRowsInfo')
        self.selected_rows_info_string_value_tag = create_tag('FileDialog', 'StringValue', 'SelectedRowsInfo')
        self.selected_rows_dict = {}
        self.directory_theme = create_tag('FileDialog', 'Theme', 'DirectoryTheme')
        self.mat_theme = create_tag('FileDialog', 'Theme', 'MatTheme')
        self.hdf5_theme = create_tag('FileDialog', 'Theme', 'HDF5Theme')
        self.nifti_theme = create_tag('FileDialog', 'Theme', 'NiftiTheme')
        self.dicom_theme = create_tag('FileDialog', 'Theme', 'DicomTheme')
        self.theme_dict = {'dir': self.directory_theme,
                           'mat': self.mat_theme,
                           'hdf5': self.hdf5_theme,
                           'dicom': self.dicom_theme,
                           'nifti': self.nifti_theme}
        # self.default_table_size = [0.56, 0.1, 0.215, 0.125]
        self.default_table_size = [0.575, 0.10, 0.20, 0.125]
        self.chosen_file_nodes = {}
        self.chosen_table_theme = create_tag('FileDialog', 'Theme', 'ChosenFilesTable')
        self.current_sort_app_data = []
        self.current_filter_app_data = []
        self.chosen_table_user_data = [0, []] # current_indent_level, [selected_rows]
        self.chosen_volume_text_tag = create_tag('FileDialog', 'Text', 'ChosenVolumes')
        self.is_windows_drive_parent = False
        self.last_selected_file = None
        self.parsed_file = None
        self.load_file_dict = None

    def __del__(self):
        if not self.has_parent:
            dpg.destroy_context()

    def bind_theme_value(self, item, file_dict):
        file_type = file_dict['File Type']
        if file_type in self.theme_dict.keys():
            dpg.bind_item_theme(item, self.theme_dict[file_type])

    def initialize(self, 
                   VolumeLayerGroups: VolumeLayer.VolumeLayerGroups = None,
                   DrawWindow: NewMainView.MainView = None, 
                   OptionsPanel: OptionsPanel.OptionsPanel = None,
                   InformationBox: InformationBox.InformationBox = None,
                   debug = False):
        if not self.has_parent:
            dpg.create_context()

        self.VolumeLayerGroups = VolumeLayerGroups
        self.DrawWindow = DrawWindow
        self.OptionsPanel = OptionsPanel
        self.InformationBox = InformationBox

        dpg.add_item_double_clicked_handler(button = dpg.mvMouseButton_Left, tag = self.double_click_handler_tag, callback = self.double_click_callback, parent = G.ITEM_HANDLER_REG_TAG)
        dpg.add_item_clicked_handler(button = dpg.mvMouseButton_Left, tag = self.single_click_handler_tag, callback = self.item_single_clicked, parent = G.ITEM_HANDLER_REG_TAG)
        dpg.add_string_value(default_value = self.selected_rows_info_string, tag = self.selected_rows_info_string_value_tag, parent = G.VALUE_REG_TAG)

        # with dpg.value_registry(tag = self.value_registry_tag):
        #     dpg.add_string_value(default_value = self.selected_rows_info_string, tag = self.selected_rows_info_string_value_tag)

        dpg.add_theme(tag = self.directory_theme)

        with dpg.theme_component(dpg.mvSelectable, parent = self.directory_theme):
            dpg.add_theme_color(dpg.mvThemeCol_Text, [220, 0, 40])

        dpg.add_theme(tag = self.mat_theme)
        with dpg.theme_component(dpg.mvSelectable, parent = self.mat_theme):
            dpg.add_theme_color(dpg.mvThemeCol_Text, [220, 220, 40])

        dpg.add_theme(tag = self.hdf5_theme)
        with dpg.theme_component(dpg.mvSelectable, parent = self.hdf5_theme):
            dpg.add_theme_color(dpg.mvThemeCol_Text, [0, 220, 40])

        dpg.add_theme(tag = self.nifti_theme)
        with dpg.theme_component(dpg.mvSelectable, parent = self.nifti_theme):
            dpg.add_theme_color(dpg.mvThemeCol_Text, [40, 220, 220])

        dpg.add_theme(tag = self.dicom_theme)
        with dpg.theme_component(dpg.mvSelectable, parent = self.dicom_theme):
            dpg.add_theme_color(dpg.mvThemeCol_Text, [40, 220, 0])

        dpg.add_theme(tag = self.chosen_table_theme)
        with dpg.theme_component(dpg.mvAll, parent = self.chosen_table_theme):
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 0)

        with dpg.window(label='File Dialog', 
                        tag = create_tag('FileDialog', 'Window', 'FileDialog'), 
                        width = -1, 
                        height = 740, 
                        show = False, 
                        menubar = False, 
                        modal = True, 
                        no_open_over_existing_popup = True) as self.file_dialog_window:
            
            with dpg.group(horizontal=True, tag = create_tag('FileDialog', 'Group', 'FileDialogContents')):
                with dpg.child_window(tag = create_tag('FileDialog', 'Window', 'CurrentDirectoryContents'), 
                                      width = 750, 
                                      height = 350):
                    with dpg.group(horizontal = True, 
                                   tag = self.navigation_buttons_group_tag):
                        
                        dpg.add_button(label = '', 
                                       width = 19, 
                                       height = 19, 
                                       user_data = self.drive_letters,
                                       tag = self.go_up_directory_button_tag, 
                                       arrow = True, 
                                       direction = dpg.mvDir_Up,
                                       callback = self.go_up_directory)

                        dpg.add_button(label = 'R', 
                                       width = 19, 
                                       height = 19, 
                                       user_data = False,
                                       tag = self.refresh_directory_button_tag,
                                       callback = self.refresh_directory_button)
                        
                        dpg.add_text(f'{self.format_displayed_path(self.current_directory)}', 
                                     tag = self.current_directory_text_tag)
                    
                    with dpg.table(tag = self.file_dialog_table_tag, 
                                   sortable = True, 
                                   sort_multi = True, 
                                   header_row = True, 
                                   freeze_rows = 1,
                                   callback = self.sort_files_callback,
                                   user_data = self.current_directory_file_dict,
                                   resizable = True, 
                                   width = -1, 
                                   height = 275, 
                                   scrollY = True):
                        
                        dpg.add_table_column(label = 'Name', tag = create_tag('FileDialog', 'TableColumn', 'Name'),
                                             width_stretch=True, init_width_or_weight = self.default_table_size[0])
                        dpg.add_table_column(label = 'Size', tag = create_tag('FileDialog', 'TableColumn', 'Size'), 
                                             no_resize = False, init_width_or_weight = self.default_table_size[1])
                        dpg.add_table_column(label = 'Last Modified', tag = create_tag('FileDialog', 'TableColumn', 'Last Modified'), 
                                             no_resize = False, init_width_or_weight = self.default_table_size[2])
                        dpg.add_table_column(label = 'Extension', tag = create_tag('FileDialog', 'TableColumn', 'Suffix'), 
                                             no_resize = False, init_width_or_weight = self.default_table_size[3])

                    self.populate_table_rows(self.file_dialog_table_tag, self.current_directory_file_dict)
                    self.bind_table_rows(G.ITEM_HANDLER_REG_TAG, self.file_dialog_table_tag)

                    with dpg.group(horizontal = True, tag = create_tag('FileDialog', 'HorizontalGroup', 'FileSelectionParams')):
                        dpg.add_combo(items = list(self.file_extension_dict.keys()), # app_data of form [selected_string]
                                      tag = self.file_filter_combobox_tag,
                                      width = -0, 
                                      default_value = list(self.file_extension_dict.keys())[0],
                                      callback = self.filter_table_results_callback) 
                        self.filter_combobox_app_data = list(self.file_extension_dict.keys())[0]
                        dpg.add_button(label = 'Deselect All', 
                                       tag = create_tag('FileDialog', 'Button', 'DeselectAll'),
                                       width=-1, callback = self.deselect_all)
                        
                if self.debug:
                    with dpg.child_window(tag = create_tag('FileDialog', 'Window', 'SelectedRowInfo'), width = 940, height = 350, 
                                          resizable_x = True, horizontal_scrollbar = True):
                        dpg.add_text(default_value = '', 
                                     source = self.selected_rows_info_string_value_tag, 
                                     tag = self.selected_rows_info_text_tag)
            with dpg.group(horizontal=True, tag = create_tag('FileDialog', 'Group', 'FileDialogChosen')):
                with dpg.child_window(tag = create_tag('FileDialog', 'Window', 'ChosenFiles'), width = 500, height = 350):
                    with dpg.table(tag = self.chosen_files_table_tag, resizable = True, width = -1, height = 275, 
                                   scrollY = True, user_data = self.chosen_table_user_data):
                        dpg.add_table_column(label = 'Name', tag = 'Chosen|Name', init_width_or_weight = 0.60)
                        dpg.add_table_column(label = 'Shape', tag = 'Chosen|Shape', init_width_or_weight = 0.25)
                        dpg.add_table_column(label = 'Type', tag = 'Chosen|Type', init_width_or_weight = 0.15)
                    dpg.bind_item_theme(self.chosen_files_table_tag, 
                                        self.chosen_table_theme)

                    with dpg.group(horizontal = True, tag = create_tag('FileDialog', 'HorizontalGroup', 'VolumeIOButtons')):
                        dpg.add_button(label = f'Load Selected', tag = create_tag('FileDialog', 'Button', 'LoadSelectedVolumes'), 
                                       callback = self.load_selected_volumes)
                # with dpg.child_window(tag = create_tag('FileDialog', 'Window', 'ChosenVolumes'), width = -1, height = 350):
                #     dpg.add_text(default_value = '', tag = self.chosen_volume_text_tag)

        if not self.has_parent:
            dpg.create_viewport(title='File Dialog', 
                                width = 1720, 
                                height = 1080, 
                                x_pos = 10, 
                                y_pos = 10)
            dpg.setup_dearpygui()
            dpg.set_primary_window(self.file_dialog_window, True)
            dpg.show_viewport()
            dpg.start_dearpygui()

            dpg.destroy_context()
    

    def make_filedialog_modal(self):
        with dpg.mutex():
            if not dpg.get_item_configuration(self.file_dialog_window)['modal']:
                dpg.hide_item(self.file_dialog_window)
                dpg.configure_item(self.file_dialog_window, modal = True)
                dpg.show_item(self.file_dialog_window)

    
    def load_landmarks(self, volume_file_id:str, volume_id: str, landmark_file: Path):
        self.load_file_dict[volume_file_id][volume_id]


    def load_selected_volumes(self):
        indent, leaves = self.chosen_table_user_data
        if len(leaves) < 1:
            return
        print(f'FileDialog Message: Loading {self.chosen_table_user_data}')
        load_file_dict = {}
        for row_leaf in leaves:
            print(f'FileDialog Message: {dpg.get_item_user_data(dpg.get_item_user_data(row_leaf)[1])[0] = }') # Gets node_user_data
            row_user_data = dpg.get_item_user_data(dpg.get_item_user_data(row_leaf)[1])
            row_id_info = row_user_data[0]
            row_attrib_dict = row_user_data[1]
            file_id = row_id_info.split('|')[0]
            file_name = row_id_info.split('|')[1]
            file_volume = row_id_info.split('|')[-1]
            if file_id in load_file_dict.keys():
                load_file_dict[file_id]['volumes'][file_volume] = {'Attributes': dict(row_attrib_dict)}
            else:
                load_file_dict[file_id] = {}
                load_file_dict[file_id]['Attributes'] = dict(self.chosen_file_nodes[file_id]['Attributes'])
                load_file_dict[file_id]['volumes'] = {file_volume: {'Attributes': dict(row_attrib_dict)}}

        # load_file_dict[file_id] = {'Attributes': {}, 
        #                            'volumes': 
        #                                      {volume_1: 
        #                                                {'Attributes': 
        #                                                              {'affine': ...
        #                                                               }
        #                                                 }
        #                                       }
        #                             }

        self.load_file_dict = dict(load_file_dict)
        data_loader = DataLoader(self.load_file_dict)
        self.hide()
        data_loader.load_selected_files(self.VolumeLayerGroups, 
                                        self.DrawWindow)
        data_loader.finalize_load_volumes(self.VolumeLayerGroups, 
                                          self.DrawWindow, 
                                          self.OptionsPanel,
                                          self.InformationBox)
        self.deselect_all()

    def get_selected_volumes(self):
        current_indent, selected_leaves = dpg.get_item_user_data(self.chosen_table_user_data)

        print('FileDialog Message: Selected volumes:')

        for row_leaf in selected_leaves:
            print(f'FileDialog Message: \t{row_leaf}, {dpg.get_item_user_data(dpg.get_item_user_data(row_leaf)[1])[0]}')

        return 

    def close(self):
        if not self.has_parent:
            dpg.destroy_context()
        else:
            self.deselect_all()
            self.hide()

    def show(self):
        dpg.show_item(self.file_dialog_window)

    def hide(self):
        dpg.hide_item(self.file_dialog_window)

    def multisort(self, x_list, sort_specs, make_copy = True, return_sorted_list = False):
        if make_copy:
            return_list = [val for val in x_list]
        else:
            return_list = x_list
    
        # Reverse sort specs so we work our way out of the sorting appropriately, 
        # where we sort the last key first. 
        
        for index, reverse in reversed(sort_specs):
            return_list.sort(key = itemgetter(index), reverse = reverse <= 0)
    
        new_indices = []
        
        for new_index, row_tuple in enumerate(return_list):
            new_indices.append(row_tuple[-2]) # internal dpg item number, row id
        
        if return_sorted_list:
            return return_list, new_indices
        else:
            return new_indices

    def sort_files_callback(self, sender, app_data, user_data):
        """
        sender: Table|FileDialog, etc
        app_data:
            no sorting -> app_data == None
            single sorting -> app_data = [[column_id, direction]]
            multi sorting -> app_data = [[column_id, direction], [column_id, direction], ... ]
            direction == +1 -> ascending
            direction == -1 -> descending
        user_data: self.current_directory_file_dict or similar
        """

        print(f'FileDialog Message: Sort files callback:\n{sender = }\n{app_data = }', flush = True)
    
        if app_data is None: 
            return

        self.current_sort_app_data = [data for data in app_data]

        sort_columns = []
        s_specs = [[8, -1]] # [File info column number, ascending/descending], starts with is_dir
        values_to_sort = []
        for child_index, child in enumerate(dpg.get_item_children(sender, 1)):
            file_id = dpg.get_item_alias(child).split('|')[2]
            c_list = [value for value in user_data[file_id].values()]
            c_list.extend([child, child_index])
            values_to_sort.append(c_list)

        for a_data in app_data:
            alias = dpg.get_item_alias(a_data[0]).split('|')[2]
            col_number = FileDialog.column_names_to_index[alias]
            
            sort_columns.append([alias, '\u2191' if a_data[1] > 0 else '\u2193'])
            s_specs.append([col_number, a_data[1]])

        logger_info = f'Sorting file dialog: {sender = }, {app_data = }, {sort_columns = }'

        sorted_indices = self.multisort(values_to_sort, s_specs, make_copy = True, return_sorted_list = False)
        
        dpg.reorder_items(sender, 1, sorted_indices)
        print('FileDialog Message: Sorting done', flush = True)
    

    def filter_table_results_callback(self, sender, app_data):
        # app_data = "Nifti Files (.nii, .nii.gz)" or other key in self.file_extension_dict
        # self.file_extension_dict[app_data][1] this is a list of files matching the appropriate extension.
        self.filter_combobox_app_data = app_data
        for table_row in dpg.get_item_children(self.file_dialog_table_tag, 1):
            # self.table_row_index_to_tag[table_row].split(self.separator)[1] returns rel_path
            # row_tag = 'TableRow|file_id|parent_table'
            # parent_table user_data has the full dict of file info
            file_id = self.table_row_index_to_tag[table_row].split(self.separator)[2]
            is_dir = self.current_directory_file_dict[file_id]['Is Dir']
            is_mount = self.current_directory_file_dict[file_id]['Is Mount']
            if (file_id in self.file_extension_dict[app_data][1]) or is_dir:
                dpg.show_item(table_row)
            else:
                dpg.hide_item(table_row)

    
    def get_table_contents(self, parent_table):
        table_contents = {}
        if len(dpg.get_item_children(parent_table, 1)) == 0:
            print(f'FileDialog Message: No children found in {parent_table}')
            return
        for row_id in dpg.get_item_children(parent_table, 1):
            table_contents[dpg.get_item_alias(row_id)] = dpg.get_item_children(row_id, 1)
            print(f'FileDialog Message: {row_id = }, {dpg.get_item_alias(row_id) = }, {dpg.get_item_children(row_id, 1) = }')


    def populate_table_rows(self, parent_table, file_dict:dict, row_type = 'selectable'):
        """
        file_dict -> None or file dictionary to add as rows.
        """

        for file_id in file_dict.keys():
            row_tag = create_tag('FileDialog', 'TableRow', f'{file_id}', suffix = parent_table)
            with dpg.table_row(tag = row_tag, 
                               parent = parent_table, 
                               user_data = (file_id, file_dict[file_id])):
                self.table_row_index_to_tag[dpg.last_item()] = row_tag
                self.table_row_tag_to_index[row_tag] = dpg.last_item()
                self.bind_theme_value(row_tag, file_dict[file_id])
                for f_info in self.file_table_columns: # [Name', 'Formatted Size', 'Last Modified', 'Extension']
                    tag = create_tag('FileDialog', 'SelectableRow', f'{file_id}', suffix = f'{parent_table}{self.separator}{f_info}')
                    dpg.add_selectable(label = f'{file_dict[file_id][f_info]}',
                                       tag = tag,
                                       user_data = (file_id, file_dict[file_id]),
                                       span_columns = True,
                                       callback = self.update_selected_rows)
    

    def bind_table_rows(self, handler, table_tag):
        # print(f'FileDialog Message: Binding items to {handler = }.')
        for row in dpg.get_item_children(table_tag, 1):
            if len(dpg.get_item_children(row, 1)) > 0:
                for row_child in dpg.get_item_children(row, 1):
                    # print(f'FileDialog Message: \tBinding {row_child = }, {dpg.get_item_alias(row_child) = }, {dpg.get_item_type(row_child) = }')
                    dpg.bind_item_handler_registry(row_child, handler)


    def reset_table_rows(self, parent_table):
        dpg.delete_item(parent_table, children_only = True, slot = 1)
    

    def item_single_clicked(self, sender, app_data, user_data):
        # print(f'FileDialog Message: Single item clicked!\n----------\n{sender = }\n{app_data = }\n{user_data = }\n----------')
        pass
                        

    def refresh_current_directory(self, sender, app_data, is_windows_drive_parent = False):
        self.reset_table_rows(self.file_dialog_table_tag)

        self.current_directory_file_dict.clear()
        self.table_row_tag_to_index.clear()
        self.table_row_index_to_tag.clear()

        self.parse_files_in_path(self.current_directory, 
                                 is_windows_drive_parent = is_windows_drive_parent) # self.current_directory_file_dict, self.file_extension_dict
        
        print('Populating Table Rows')
        self.populate_table_rows(self.file_dialog_table_tag, self.current_directory_file_dict)
        print('Binding Table Rows')
        self.bind_table_rows(G.ITEM_HANDLER_REG_TAG, self.file_dialog_table_tag)

        for file_id in self.selected_rows_dict.keys():
            for table_row in self.table_row_tag_to_index.keys():
                if file_id in table_row:
                    dpg_id = self.table_row_tag_to_index[table_row]
                    # We only need the first column since that selectable covers the entire row. 
                    for row_child in dpg.get_item_children(dpg_id, 1)[:1]:
                        dpg.set_value(row_child, True)
        
        print('Filter Callback Table Rows')
        self.filter_table_results_callback(self.file_filter_combobox_tag, dpg.get_value(self.file_filter_combobox_tag))
        
        print('Sort Files Callback')
        self.sort_files_callback(self.file_dialog_table_tag, self.current_sort_app_data, dpg.get_item_user_data(self.file_dialog_table_tag))
        
        print(f'FileDialog Message: Directory {self.current_directory} refreshed.', flush = True)
        print(f'\t{self.current_directory}')

    def double_click_callback(self, sender, app_data):
        if not dpg.is_item_shown(self.file_dialog_window):
            return
        
        with dpg.mutex():
            # Gets parent 
            user_data = dpg.get_item_user_data(app_data[1])[1]
            if sender == self.go_up_directory_button_tag:
                self.change_current_directory(sender, app_data, user_data)
                self.refresh_current_directory(None, None, self.is_windows_drive_parent)
                print('FileDialog Message: double_click_callback', flush = True)
                print(f'\t{sender = }')

            elif user_data['Is Dir']:
                self.change_current_directory(sender, app_data, user_data)
                self.refresh_current_directory(None, None, self.is_windows_drive_parent)
                print('FileDialog Message: double_click_callback', flush = True)
                print(f'\t{user_data['Is Dir'] = }', flush = True)
                
            else:
                return
            

    def go_up_directory(self, sender, app_data, user_data):
        with dpg.mutex():
            self.change_current_directory(sender, app_data, user_data)
            self.refresh_current_directory(None, None, self.is_windows_drive_parent)


    def refresh_directory_button(self, sender, app_data, user_data):
        with dpg.mutex():
            self.refresh_current_directory(sender, app_data, user_data)


    def change_current_directory(self, sender, app_data, user_data):
        if sender == self.go_up_directory_button_tag:
            print(f'FileDialog Message: \tChanging directory to {self.current_directory.parent}.')
            self.current_directory = self.current_directory.parent
            
            if len(self.drive_letters) > 0:
                print(self.current_directory.as_posix())
                if self.current_directory.as_posix() in self.drive_letters: 
                    self.is_windows_drive_parent = True
                    self.current_directory = Path('WindowsDrives')
            else:
                self.is_windows_drive_parent = False

        else:
            item_tag = dpg.get_item_alias(app_data[1])
            print(f'FileDialog Message: \tChanging directory to {user_data['File']}.', flush = True)
            if user_data['Is Dir']:
                self.current_directory = Path(user_data['File'])
            else:
                pass
        
        displayed_path = self.format_displayed_path(self.current_directory, max_length=415, scale_factor=7.0)

        dpg.set_value(self.current_directory_text_tag, f'{displayed_path}')

        if not self.is_windows_drive_parent:
            G.CONFIG_DICT['directories']['image_dir'] = Path(self.current_directory).as_posix()

    def format_displayed_path(self, path:Path, max_length = 415, scale_factor = 7.0):
        
        displayed_path = '.../'
        displayed_text_length = scale_factor*len(displayed_path)
        parts_list = []

        for part in reversed(list(path.parts)):
            if displayed_text_length + scale_factor*(len(part) + 1) >= max_length:
                break
            parts_list.append(part)
            displayed_text_length += scale_factor*(len(part) + 1)
            
        displayed_path = Path(displayed_path)
        for part in reversed(parts_list):
            displayed_path = displayed_path.joinpath(part)

        return displayed_path.as_posix()

    def update_selected_rows(self, sender, app_data, user_data):
        """
        sender is of form: 'SelectableRow|562949953663636|Name'
        app_data is of form: True
        user_data is of form: (file_id, current_directory_file_dict[file_id])
        """
        self.selected_rows_info_string = ''

        file_id, file_dict = user_data

        # if file_dict['File Type'] == 'dir':
        #     dpg.set_value(sender, False)
        #     return

        if app_data and (file_id not in self.selected_rows_dict.keys()):
            self.selected_rows_dict[file_id] = file_dict

        if (not app_data) and (file_id in self.selected_rows_dict.keys()):
            self.selected_rows_dict.pop(file_id, None)
            self.selected_files_dict.pop(file_id, None)
        
        for selected_row_key in self.selected_rows_dict.keys():
            formatted_file_info = FileDialog.format_file_info_string(self.selected_rows_dict[selected_row_key])
            self.selected_rows_info_string = f'{self.selected_rows_info_string}{"-"*50}{formatted_file_info}\n'

        dpg.set_value(self.selected_rows_info_string_value_tag, self.selected_rows_info_string)

        self.update_selected_files(self.selected_rows_dict)
        self.update_chosen_file_nodes(self.selected_files_dict)
        self.populate_chosen_rows(self.chosen_files_table_tag, self.chosen_file_nodes)
        self.chosen_files_table_user_data = dpg.get_item_user_data(self.chosen_files_table_tag)


    def update_selected_files(self, selected_row_dict:dict):
        for key, value in selected_row_dict.items():
            if (key not in self.selected_files_dict.keys()) and (value['Is File'] or (value['File Type'] == 'dicom_dir')):
                self.selected_files_dict[key] = value

    
    def update_chosen_file_nodes(self, selected_file_dict:dict):
        file_dict = {}
        for key, value in selected_file_dict.items():
            file_dict[value['File ID']] = value['Name']
            if value['File ID'] not in self.chosen_file_nodes.keys():
                if value['File Type'] in ['mat','hdf5','dicom','nifti','dicom_dir']:
                    self.last_selected_file = FileParser(value)
                    # print(self.last_selected_file.print())
                    self.chosen_file_nodes[value['File ID']] = self.last_selected_file.file_contents

            else: 
                pass
            
        chosen_keys = list(self.chosen_file_nodes.keys())
        for key in chosen_keys:
            if key not in list(file_dict.keys()):
                print(f'FileDialog Message: Deleting {key} at row {self.chosen_file_nodes[key]["Row"]}')
                delete_node(self.chosen_files_table_tag, self.chosen_file_nodes[key]['Row'])
                del self.chosen_file_nodes[key]


    def populate_chosen_rows(self, parent_table: int|str, node_dict:dict):
        for file_node in node_dict.keys():
            if 'Row' in node_dict[file_node].keys():
                continue
            else:
                r_list = []
                #self.chosen_file_nodes[value['File ID']] = self.last_selected_file.file_contents
                create_tree_node(node_dict[file_node], 
                                 parent_table = parent_table, 
                                 row_list = r_list, 
                                 add_popup=True, 
                                 node_user_data = [f'{file_node}', node_dict[file_node]['Attributes']])


    def deselect_all(self):
        self.selected_rows_dict.clear()
        self.selected_files_dict.clear()
        self.update_chosen_file_nodes(self.selected_files_dict)

        self.selected_rows_info_string = ''

        for row_id in dpg.get_item_children(self.file_dialog_table_tag, 1):
            for row_child in dpg.get_item_children(row_id, 1):
                dpg.set_value(row_child, False)

        for row_id in dpg.get_item_children(self.chosen_files_table_tag, 1):
            dpg.delete_item(row_id, slot = 1)

        dpg.set_value(self.selected_rows_info_string_value_tag, self.selected_rows_info_string)

    
    def get_file_info(self, filepath: Path, format_file_info = True, exclude = ['Size']):
        """
        Exclude can be:
            'File'
            'Parent'
            'Grandparent'
            'Relative Path'
            'Permissions'
            'Size'
            'Last Accessed'
            'Last Modified'
            'Creation Time'
        """
        f_info_dict = {}
        for f_info in FileDialog.file_info_object.keys():
            f_info_dict[f_info] = FileDialog.file_info_object[f_info](filepath)

        if type(f_info_dict['Dicom Dir']) == type(dict()):
            f_info_dict['File Type'] = 'dicom_dir'

        if format_file_info:
            for f_info in FileDialog.file_info_formatter.keys():
                if f_info in exclude:
                    continue
                else:
                    f_info_dict[f_info] = FileDialog.file_info_formatter[f_info](f_info_dict[f_info])
            
        return f_info_dict
    
    
    def parse_files_in_path(self, path: Path, is_windows_drive_parent: bool = False):
        
        if is_windows_drive_parent:
            for drive_letter in self.drive_letters:
                drive_path = Path(drive_letter)
                drive_id = FileDialog.get_file_id(drive_path)
                self.current_directory_file_dict[f'{drive_id}'] = self.get_file_info(drive_path)
                self.current_directory_file_dict[f'{drive_id}']['Name'] = drive_letter[:-1]

        else:
            for file in path.glob('*'):
                print(f'FileDialog Message: FILE_PARSER: {file.name}', flush = True)
                file_id = FileDialog.get_file_id(file)
                self.current_directory_file_dict[f'{file_id}'] = self.get_file_info(file, format_file_info = True, exclude = ['Size'])
                match self.current_directory_file_dict[f'{file_id}']['Suffix']:
                    case '.mat':
                        self.file_extension_dict["MATLab Files (.mat)"][1].append(f'{file_id}')
                        self.file_extension_dict["Image Files (.mat, .dcm, .h5, .hdf5, .nii, .nii.gz)"][1].append(f'{file_id}')
                    case '.dcm':
                        self.file_extension_dict["DICOM Files (.dcm)"][1].append(f'{file_id}')
                        self.file_extension_dict["Image Files (.mat, .dcm, .h5, .hdf5, .nii, .nii.gz)"][1].append(f'{file_id}')
                    case '.h5' | '.hdf5':
                        self.file_extension_dict["HDF5 Files (.h5, .hdf5)"][1].append(f'{file_id}')
                        self.file_extension_dict["Image Files (.mat, .dcm, .h5, .hdf5, .nii, .nii.gz)"][1].append(f'{file_id}')
                    case '.nii' | '.nii.gz': 
                        self.file_extension_dict["Nifti Files (.nii, .nii.gz)"][1].append(f'{file_id}')
                        self.file_extension_dict["Image Files (.mat, .dcm, .h5, .hdf5, .nii, .nii.gz)"][1].append(f'{file_id}')
        
        self.file_extension_dict["All Files (*)"][1].extend(list(self.current_directory_file_dict.keys()))
        
        return


class FileParser(object):
    default_dict = {'Name': 'DEFAULT', 'Type': '', 'Data': {}, 'Attributes': {'DEFAULT': 'DEFAULT'}}

    def __init__(self, file_info_object, suffix_4d = '__'):

        """
        file_info_object = {'File': lambda x: x,
                            'Parent': lambda x: x.parent,
                            'Grandparent': lambda x: x.parent.parent,
                            'Relative Path': lambda x: x.relative_to(x.parent.parent),
                            'Name': lambda x: x.name,
                            'Stem': lambda x: x.stem,
                            'Suffix': lambda x: FileDialog.parse_file_suffix(x),
                            'Is File': lambda x: x.is_file(),
                            'Is Dir': lambda x: x.is_dir(),
                            'Permissions': lambda x: x.stat().st_mode,
                            'File ID': lambda x: x.stat().st_ino,
                            'Size': lambda x: x.stat().st_size,
                            'Formatted Size': lambda x: FileDialog.format_file_size(x.stat().st_size),
                            'Last Accessed': lambda x: (x.stat().st_atime, datetime.datetime.fromtimestamp(np.round(x.stat().st_atime))),
                            'Last Modified': lambda x: (x.stat().st_mtime, datetime.datetime.fromtimestamp(np.round(x.stat().st_mtime))),
                            'Creation Time': lambda x: (x.stat().st_birthtime, datetime.datetime.fromtimestamp(np.round(x.stat().st_birthtime))),
                            'File Type': lambda x: FileDialog.determine_file_type(x)}

        file_info_formatter = {'File': lambda x: x.as_posix(),
                               'Parent': lambda x: x.as_posix(),
                               'Grandparent': lambda x: x.as_posix(),
                               'Relative Path': lambda x: x.as_posix(),
                               'Permissions': lambda x: oct(x)[2:],
                               'Size': lambda x: FileDialog.format_file_size(x), 
                               'Last Accessed': lambda x: f'{x[1]}',
                               'Last Modified': lambda x: f'{x[1]}',
                               'Creation Time': lambda x: f'{x[1]}'}
        """

        self.parse_dict = {'mat': self.parse_mat_file,
                           'hdf5': self.parse_hdf5_file,
                           'dicom': self.parse_dicom_files,
                           'dicom_dir': self.parse_dicom_dir,
                           'nifti': self.parse_nifti_file,
                           'landmark': self.parse_landmark_file}
        
        self.matfile_version_dict = {'(0, 0)': ['v4', 'mat'],
                                     '(1, 0)': ['v5', 'mat'],
                                     '(2, 0)': ['v7.3', 'h5']}

        # We want a new dictionary made from the old one, not just a pointer. 
        self.file_info_object = dict(file_info_object) 
        self.file_contents = {'Name': file_info_object['Name'], 'Type': 'Group', 'Data': {}, 'Attributes': {'four_d': False, 
                                                                                                            'file_type': self.file_info_object['File Type'],
                                                                                                            'file_path': Path(self.file_info_object['File']),
                                                                                                            'file_id': self.file_info_object['File ID'],
                                                                                                            'file_name': self.file_info_object['Name']}}
        self.array4D = False
        self.suffix_4d = suffix_4d
        self.parse_file(self.file_info_object, self.file_contents)

    def print(self):
        print(self.file_contents)

    def parse_file(self, file_info_object, file_contents):

        return self.parse_dict[file_info_object['File Type']](file_info_object, file_contents)

    def parse_mat_file(self, file_info_object: dict, file_content_dict: dict = None): 
        mat_version, updated_file_type = self.matfile_version_dict[f'{matfile_version(file_info_object["File"])}']
        if updated_file_type == 'hdf5':
            self.parse_hdf5_file(file_info_object, file_content_dict)
            return

        mat_variables = whosmat(file_info_object['File'])

        if type(file_content_dict) == type(None):
            file_content_dict = dict(FileParser.default_dict)

        for var_name, var_shape, var_dtype in mat_variables:
            if len(var_shape) <= 3:
                file_content_dict['Data'][var_name] = {'Name': var_name, 'Type': 'Dataset', 'Attributes': {'shape': var_shape, 'dtype': var_dtype}}
            
            elif len(var_shape) == 4:
                file_content_dict['Attributes']['four_d'] = True
                for index in range(var_shape[0]):
                    zfilled_index = f'{index}'.zfill(len(f'{var_shape[0]}'))
                    name = f'{var_name}{self.suffix_4d}{zfilled_index}'
                    file_content_dict['Data'][var_name] = {'Name': name, 'Type': 'Dataset', 'Attributes': {'shape': var_shape[1:], 'dtype': var_dtype}}

            else: 
                pass
        
        if type(file_content_dict) == type(dict()):
            return file_content_dict


    def parse_hdf5_file(self, file_info_object: dict, file_content_dict: dict = None):

        with h5py.File(file_info_object['File'], mode = 'r', swmr = True, track_order = True) as hdf5_stream:
            self.retrieve_hdf5_info(hdf5_stream, file_content_dict = file_content_dict)

        if type(file_content_dict) == type(None):
            return file_content_dict


    def retrieve_hdf5_info(self, hdf5_object:h5py.Group|h5py.Dataset, file_content_dict:dict = None):
        """
        Recursively finds all information in an hdf5 file and stores it in a hierarchical dictionary.
        
        Uses retrieve hdf5_attrs.
        """

        return_dict = False
        if type(file_content_dict) == type(None):
            file_content_dict = {'Name': hdf5_object.file.filename, 'Type': 'Group', 'Data': {}, 'Attributes': self.retrieve_hdf5_attrs(hdf5_object)}
            return_dict = True

        else:
            file_content_dict['Attributes'].update(self.retrieve_hdf5_attrs(hdf5_object))
                
        if isinstance(hdf5_object, h5py.Group):
            file_content_dict['Type'] = 'Group'
            for key, value in hdf5_object.items():
                if isinstance(value, h5py.Group):
                    file_content_dict['Data'][key] = {'Name': key, 'Type': 'Group', 'Data': {}, 'Attributes': {}}
                    self.retrieve_hdf5_info(value, file_content_dict = file_content_dict['Data'][key])
                
                else:
                    attrs = self.retrieve_hdf5_attrs(value)
                    attrs['shape'] = value.shape
                    attrs['dtype'] = value.dtype
                    if len(value.shape) == 4:
                        self.array4D = True
                        attrs['shape'] = value.shape[1:]
                        for index in range(len(value)):
                            zfilled_index = f'{index}'.zfill(len(f'{value.shape[0]}'))
                            name = f'{value.name}{self.suffix_4d}{zfilled_index}'
                            file_content_dict['Data'][name] = {'Name': name,
                                                               'Type': 'Dataset',
                                                               'Attributes': attrs}

                    else:
                        file_content_dict['Data'][key] = {'Name': value.name, 
                                                          'Type': 'Dataset',
                                                          'Attributes': attrs}



        elif isinstance(hdf5_object, h5py.Dataset):
            file_content_dict['Type'] = 'Dataset'
        
        if return_dict:
            return file_content_dict

    def retrieve_hdf5_attrs(self, hdf5_object:h5py.Group|h5py.Dataset):
        """
        Function to retrieve attribute keys and values and return them as a dict. 
        """
        
        attrs_dict = {}
        
        for key, value in hdf5_object.attrs.items():
            attrs_dict[key] = value
            
        return attrs_dict

    def parse_dicom_files(self, file_info_object, file_content_dict):

        if type(file_content_dict) == type(None):
            file_content_dict = dict(FileParser.default_dict)

        if type(file_info_object['Dicom Dir']) != type(None):
            self.parse_dicom_dir(file_info_object, file_content_dict)

    def parse_dicom_dir(self, directory_info_object, file_content_dict):
        """
        dcm_dir_contents[f'{seriesUID}'].append(file)
        """
        for seriesUID in list(directory_info_object['Dicom Dir'].keys()):
            series_shape = directory_info_object['Dicom Dir'][seriesUID]['shape']
            series_dtype = directory_info_object['Dicom Dir'][seriesUID]['dtype']
            series_affine = directory_info_object['Dicom Dir'][seriesUID]['affine'][:]
            series_im_pos = directory_info_object['Dicom Dir'][seriesUID]['im_pos'][:]
            series_dir = directory_info_object['Dicom Dir'][seriesUID]['dir']
            print(f'FileParser Message: Parsing {seriesUID}\n\t\t\t\tShape:{series_shape}.\n\t\t\t\tData Type:{series_dtype}')
            file_content_dict['Data'][seriesUID] = {'Name': seriesUID, 'Type': 'Dataset', 'Attributes': {'shape': series_shape, 
                                                                                                         'dtype': series_dtype, 
                                                                                                         'affine': series_affine,
                                                                                                         'im_pos': series_im_pos,
                                                                                                         'dir': series_dir}}


    def parse_nifti_file(self, file_info_object, file_content_dict):

        if type(file_content_dict) == type(None):
            file_content_dict = dict(FileParser.default_dict)

        nib_file = nib.load(file_info_object['File'])
        var_affine = nib_file.affine
        var_shape = nib_file.shape
        var_dtype = nib_file.header.get_data_dtype()

        file_content_dict['Data']['volume'] = {'Name': 'volume', 'Type': 'Dataset', 'Attributes': {'shape': var_shape, 'affine': var_affine, 'dtype': var_dtype}}

        if type(file_content_dict) == type(dict()):
            return file_content_dict

    def parse_landmark_file(self, file_info_object, file_content_dict):
        pass

class DataLoader(object):
    """
    The purpose of this class is to pass the selected files and data to G.APP.Volumes and G.APP.VolumeLayerGroups using CTVolume.CTVolume. 
    """
    def __init__(self, files_to_be_loaded_dict):
        
        self.load_type_dict = {'mat': self.load_mat_file,
                               'hdf5': self.load_hdf5_file,
                               'dicom': self.load_dicom_files,
                               'nifti': self.load_nifti_file,
                               'dicom_dir': self.load_dicom_files}
        
        self.files_to_be_loaded_dict = dict(files_to_be_loaded_dict)
        print(f'DataLoader Message: Initialized DataLoader')
        if dpg.does_item_exist(G.LOADING_WINDOW_TAG) or dpg.does_alias_exist(G.LOADING_WINDOW_TAG):
            return
        
        else:
            self.create_loading_window()
        


    def finalize_load_volumes(self, 
                              VolumeLayerGroups: VolumeLayer.VolumeLayerGroups,
                              DrawWindow: NewMainView.MainView,
                              OptionsPanel: OptionsPanel.OptionsPanel,
                              InformationBox: InformationBox.InformationBox):
        

        if VolumeLayerGroups.get_group_by_index(0).n_volumes > 0:
            if not VolumeLayerGroups.active:
                VolumeLayerGroups.set_current_volume_by_index(0, 0)
                VolumeLayerGroups.get_current_volume().add_textures_to_drawlists(VolumeLayerGroups.texture_drawlayer_tags)
                # VolumeLayerGroups.get_current_volume().add_texture_to_drawlist(drawlist=DrawWindow.return_texture_drawlayer_tag(window_tag))
                VolumeLayerGroups.get_current_volume().set_colormap_scale_tag(DrawWindow.return_colormap_tag(DrawWindow.get_window_tags()[0]))
                VolumeLayerGroups.get_current_group().set_colormap_scale_tag(DrawWindow.return_colormap_tag(DrawWindow.get_window_tags()[0]))
                # VolumeLayerGroups.get_current_group().set_drawlayer_tags(DrawWindow.get_texture_drawlist_tags())

                InformationBox.load_image(VolumeLayerGroups)

                VolumeLayerGroups.set_active()

        G.N_VOLUMES = len(VolumeLayerGroups.get_current_group().volume_names)
        G.OPTIONS_DICT['img_index_slider']['max_value'] = VolumeLayerGroups.get_current_group().n_volumes
        dpg.set_item_user_data(G.OPTIONS_DICT['img_index_slider']['slider_tag'],
                               VolumeLayerGroups.current_group_and_volume)
        
        dpg.configure_item(G.OPTIONS_DICT['img_index_slider']['slider_tag'], 
                            max_value = VolumeLayerGroups.get_group_by_name('AllVolumes').n_volumes)
        
        G.APP.ImageTools.update_selector_lists(VolumeLayerGroups.get_current_group().volume_names)
        G.APP.ImageTools.enable_options()

        InformationBox.initialize_landmark_tables(VolumeLayerGroups.get_current_group().volume_names)
        
        OptionsPanel.enable_options()
        dpg.set_item_label(G.VOLUME_TAB_TAG, f'Volume Tab: {VolumeLayerGroups.get_current_volume().name}')
        
        OptionsPanel.update_volume('FileDialog', None, None)
        # VolumeLayerGroups.get_current_group().set_landmark_draw_layer_tag(DrawWindow.return_landmark_drawlayer_tag(window_tag))
        VolumeLayerGroups.update_histogram('volume')
        VolumeLayerGroups.update_histogram('texture')

        affine = VolumeLayerGroups.get_volume_by_index(0, 0).CTVolume.affine
        layers_tab_text = f'{VolumeLayerGroups.get_volume_by_index(0, 0).name}'

        for row in affine:
            layers_tab_text = f'{layers_tab_text}\n\t{row}'

        for vol_index in range(1, VolumeLayerGroups.get_group_by_index(0).n_volumes):
            affine = VolumeLayerGroups.get_volume_by_index(0, vol_index).CTVolume.affine
            vol_name = VolumeLayerGroups.get_volume_by_index(0, vol_index).name
            layers_tab_text = f'{layers_tab_text}\n{vol_name}'
            
            for row in affine:
                layers_tab_text = f'{layers_tab_text}\n\t{row}'

        dpg.set_value('InfoBoxTab_layers_text', layers_tab_text)
        self.hide_loading_window()

    def show_loading_window(self):
        dpg.show_item(G.LOADING_WINDOW_TAG)

    def hide_loading_window(self):
        dpg.hide_item(G.LOADING_WINDOW_TAG)
        dpg.set_value(G.LOADING_WINDOW_TEXT, '')

    def create_loading_window(self):
        with dpg.window(label="Loading Volume", 
                        modal=False, 
                        show=False, 
                        tag=G.LOADING_WINDOW_TAG,
                        width = 800, 
                        height = 400, 
                        pos = [250, 250]):
            dpg.add_text('', tag=G.LOADING_WINDOW_TEXT)

    def load_selected_files(self, 
                            VolumeLayerGroups: VolumeLayer.VolumeLayerGroups, 
                            DrawWindow: NewMainView.MainView,
                            files_to_be_loaded_dict = None):
        
        self.show_loading_window()

        if type(files_to_be_loaded_dict) == type(None):
            files_to_be_loaded_dict = dict(self.files_to_be_loaded_dict)

        for file_id in list(files_to_be_loaded_dict.keys()):
            
            file_name = files_to_be_loaded_dict[file_id]['Attributes']['file_name']
            file_type = files_to_be_loaded_dict[file_id]['Attributes']['file_type']

            load_message = f'Loading {file_name}\n\tFile ID: {file_id}'
            dpg.set_value(G.LOADING_WINDOW_TEXT, load_message)

            VolumeLayerGroups.add_volume_to_group('AllVolumes',
                                                  self.load_type_dict[file_type](files_to_be_loaded_dict, 
                                                                                 file_id, 
                                                                                 file_name = file_name))

    def load_mat_file(self, files_to_be_loaded_dict, file_id, file_name = '') -> CTVolume.CTVolume:
        for volume_name in files_to_be_loaded_dict[file_id]['volumes']:
            if file_name == '':
                file_name = files_to_be_loaded_dict[file_id]['Attributes']['file_name']
            print(f'FileDialog Message: MatFile Load: {file_name}|{volume_name}')
            if volume_name[0] == '/':
                volume_name = volume_name[1:]
            if len(files_to_be_loaded_dict[file_id]['volumes']) == 1:
                display_name = f'{file_name}'
            else:
                display_name = f'{file_name}/{volume_name}'

            return CTVolume.CTVolume(display_name, 
                                     files_to_be_loaded_dict[file_id]['Attributes']['file_path'],
                                     loadmat(files_to_be_loaded_dict[file_id]['Attributes']['file_path'], 
                                     appendmat = False, 
                                     variable_names = volume_name, 
                                     squeeze_me = True)[volume_name],
                                     affine = np.eye(4, dtype = np.float32),
                                     dim_order = (0, 1, 2))
        

    def load_hdf5_file(self, files_to_be_loaded_dict, file_id, file_name = '') -> CTVolume.CTVolume:
        with h5py.File(files_to_be_loaded_dict[file_id]['Attributes']['file_path'], mode = 'r', swmr=True, track_order=True) as h5_file:
            for volume_name in files_to_be_loaded_dict[file_id]['volumes']:
                if file_name == '':
                    file_name = files_to_be_loaded_dict[file_id]['Attributes']['file_name']
                print(f'FileDialog Message: HDF5 Load: {file_name}|{volume_name}')
                affine = np.eye(4, 4)
                if 'affine' in files_to_be_loaded_dict[file_id]['volumes'][volume_name]['Attributes'].keys():
                    affine = files_to_be_loaded_dict[file_id]['volumes'][volume_name]['Attributes']['affine']
                if volume_name[0] == '/':
                    volume_name = volume_name[1:]
                if len(files_to_be_loaded_dict[file_id]['volumes']) == 1:
                    display_name = f'{file_name}'
                else:
                    display_name = f'{file_name}/{volume_name}'

                return CTVolume.CTVolume(display_name, 
                                         files_to_be_loaded_dict[file_id]['Attributes']['file_path'],
                                         h5_file[volume_name][()],
                                         affine = affine,
                                         dim_order = (0, 1, 2))
                

    def load_dicom_files(self, files_to_be_loaded_dict, file_id, file_name = '') -> CTVolume.CTVolume:
        for volume_name in files_to_be_loaded_dict[file_id]['volumes']:
            
            if file_name == '':
                file_name = files_to_be_loaded_dict[file_id]['Attributes']['file_name']
            print(f'DataLoader Message: DICOM Load: {file_name}|{volume_name}')
            file_path = files_to_be_loaded_dict[file_id]['Attributes']['file_path']
            affine = files_to_be_loaded_dict[file_id]['volumes'][volume_name]['Attributes']['affine']

            print(f'DataLoader Message: File {file_name} Affine:\n\t{affine}')
            if len(files_to_be_loaded_dict[file_id]['volumes']) == 1:
                display_name = f'{file_name}'
            else:
                display_name = f'{file_name}/{volume_name}'

            return CTVolume.CTVolume(display_name, 
                                     file_path,
                                     self.read_dicom_pixel_dir(file_path),
                                     affine = affine,
                                     dim_order = (0, 1, 2))
        

    def load_nifti_file(self, files_to_be_loaded_dict, file_id, file_name = '') -> CTVolume.CTVolume:
        """
        Nifti files store their data using RAS+, in contrast to DICOMs LPS+. 
        Additionally, Nifti's use IJK storage, rather than XYZ. That means the rows and columns are swapped. 
        We load DICOMS as ZXY, though, so we need to move the axes around and flip them. 
        """
        for volume_name in files_to_be_loaded_dict[file_id]['volumes']:
            if file_name == '':
                file_name = files_to_be_loaded_dict[file_id]['Attributes']['file_name']
            print(f'FileDialog Message: Nifti Load: {file_name}|{volume_name}')
            nib_file = nib.load(files_to_be_loaded_dict[file_id]['Attributes']['file_path'])
            if volume_name[0] == '/':
                volume_name = volume_name[1:]

            if len(files_to_be_loaded_dict[file_id]['volumes']) == 1:
                display_name = f'{file_name}'
            else:
                display_name = f'{file_name}/{volume_name}'
            return CTVolume.CTVolume(display_name, 
                                     files_to_be_loaded_dict[file_id]['Attributes']['file_path'],
                                     np.array(nib_file.get_fdata(), dtype = np.float32),
                                     
                                     affine = nib_file.affine,
                                     dim_order = (0, 1, 2))
                                     # dim_order = (2, 1, 0))


    # Dicom Utilities
    def convert_gdcm_type_to_numpy(self, gdcm_type):
        # gdcm_to_numpy_dtypes = {
        #     gdcm.PixelFormat.UINT8: np.uint8,     # Unsigned 8-bit integer
        #     gdcm.PixelFormat.UINT16: np.uint16,   # Unsigned 16-bit integer
        #     gdcm.PixelFormat.UINT32: np.uint32,   # Unsigned 32-bit integer
        #     gdcm.PixelFormat.UINT64: np.uint64,   # Unsigned 64-bit integer
        #     gdcm.PixelFormat.INT8: np.int8,       # Signed 8-bit integer
        #     gdcm.PixelFormat.INT16: np.int16,     # Signed 16-bit integer
        #     gdcm.PixelFormat.INT32: np.int32,     # Signed 32-bit integer
        #     gdcm.PixelFormat.FLOAT32: np.float32, # 32-bit floating point
        #     gdcm.PixelFormat.FLOAT64: np.float64, # 64-bit floating point
        #     gdcm.PixelFormat.SINGLEBIT: np.bool_, # Single bit (boolean)
        #     gdcm.PixelFormat.UINT12: np.uint16,   # 12-bit integer, stored in 16 bits
        # }
        gdcm_to_numpy_dtypes = {
        'UINT8': np.uint8,     # Unsigned 8-bit integer
        'UINT16': np.uint16,   # Unsigned 16-bit integer
        'UINT32': np.uint32,   # Unsigned 32-bit integer
        'UINT64': np.uint64,   # Unsigned 64-bit integer
        'INT8': np.int8,       # Signed 8-bit integer
        'INT16': np.int16,     # Signed 16-bit integer
        'INT32': np.int32,     # Signed 32-bit integer
        'FLOAT32': np.float32, # 32-bit floating point
        'FLOAT64': np.float64, # 64-bit floating point
        'SINGLEBIT': np.bool_, # Single bit (boolean)
        'UINT12': np.uint16,   # 12-bit integer, stored in 16 bits
        }
        return gdcm_to_numpy_dtypes[gdcm_type]

    def retrieve_image_info(self, file, gdcm_reader = gdcm.ImageReader()):
        """
        Returns rows, columns, buffer_length, pixel_format, pixel_samples, pixel_nbytes
        """
        gdcm_reader.SetFileName(f'{file}')
        if not gdcm_reader.Read():
            print(f'Cannot read {file}')
            return
            
        rows, columns = gdcm_reader.GetImage().GetDimensions()
        buffer_length = gdcm_reader.GetImage().GetBufferLength()
        pixel_format = gdcm_reader.GetImage().GetPixelFormat().GetScalarTypeAsString()
        pixel_samples = gdcm_reader.GetImage().GetPixelFormat().GetSamplesPerPixel()
        pixel_nbytes = gdcm_reader.GetImage().GetPixelFormat().GetPixelSize()

        return rows, columns, buffer_length, pixel_format, pixel_samples, pixel_nbytes

    def read_dicom_pixel_data(self, dicom_file, out_array = None, reader:gdcm.ImageReader = gdcm.ImageReader()):
        reader.SetFileName(f'{dicom_file}')
        reader.Read()
        image = reader.GetImage()
        dims = image.GetDimensions()
        
        if type(out_array) == type(None):
            return np.frombuffer(image.GetBuffer().encode('latin1', errors = "surrogateescape"), 
                                dtype = np.uint16).reshape(dims)
        else:
            out_array[:] = np.frombuffer(image.GetBuffer().encode('latin1', errors = "surrogateescape"), dtype = out_array.dtype).reshape(dims)

    # Function to process a single file in parallel
    def process_dicom_file(self, index, d_file, out_volume):
        self.read_dicom_pixel_data(d_file, out_array=out_volume[:, :, index])

    def read_dicom_pixel_dir(self, dicom_dir:Path, out_array = None):
        dicom_files = sorted(list(dicom_dir.glob('*.dcm')))
        tags = [gdcm.Tag(0x7fe0,0x0010),    # Pixel Data
                gdcm.Tag(0x0028,0x0010),    # Rows
                gdcm.Tag(0x0028,0x0011),    # Columns
                gdcm.Tag(0x0018,0x0050),    # Slice thickness
                gdcm.Tag(0x0018,0x0088),    # Space between slices
                gdcm.Tag(0x0020,0x0032),    # Image position (patient)
                gdcm.Tag(0x0020,0x0037),    # Image Orientation (patient)
                gdcm.Tag(0x0020,0x1041),    # Slice Location
                gdcm.Tag(0x0028,0x0030)]    # Pixel Spacing

        rows, columns, buffer_length, pixel_format, pixel_samples, pixel_nbytes = self.retrieve_image_info(dicom_files[0])

        return_array = False
        
        if type(out_array) == type(None):
            out_array = np.zeros((rows, columns, len(dicom_files)), dtype = self.convert_gdcm_type_to_numpy(pixel_format))
            return_array = True

        else:
            assert out_array.shape == tuple((rows, columns, len(dicom_files)))

        for file_index, file in enumerate(dicom_files):
            self.read_dicom_pixel_data(file, out_array = out_array[:, :, file_index], reader = gdcm.ImageReader())
        
        if return_array:
            return out_array

    # Function to process a single file in parallel
    def process_dicom_dir(self, dir_index, directory, out_array):
        self.read_dicom_pixel_dir(directory, out_array=out_array[dir_index])