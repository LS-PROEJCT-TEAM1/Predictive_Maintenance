"""LS semantic palette shared by Plotly and Dash components.

CSS counterparts live in the :root token block of assets/theme.css.
Selection is navy, alarms are red; numeric intensity never implies a diagnosis.
"""
BLUE = '#0A1E5A'
RED = '#FA002D'
CYAN = '#009BB4'
LIGHT = '#0569A0'
GRAY = '#7D8282'
TEXT = '#172033'
MUTED = '#475569'
BORDER = '#E7EAF0'
SURFACE = '#FFFFFF'
BLUE_SOFT = '#F0F3FA'
TEAL_SOFT = '#EDF9FA'
RED_SOFT = '#FFF1F3'
WARNING = '#AD6D00'
FRAME = '#D8DDE6'
FRAME_EDGE = '#A9B4C4'
CELL_LOW = '#C8D6E8'
MEASUREMENT_SCALE = [[0, CELL_LOW], [.5, LIGHT], [1, BLUE]]
STATUS_COLORS = {'normal': CYAN, 'warning': WARNING, 'danger': RED, 'missing': GRAY}
BLUE_SHADES = [BLUE_SOFT, '#DCE4F3', '#B8C8E5', '#8EA6D3', '#607FBF', '#3358AA', BLUE, '#08184A', '#061239', '#040C29']
