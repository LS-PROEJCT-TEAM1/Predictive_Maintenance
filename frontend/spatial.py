"""Data-linked logical inspection models. Coordinates are illustrative, not CAD."""
import math
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from dash import dcc, html
from frontend.components import BLUE, CYAN, RED, TEXT
from frontend.palette import STATUS_COLORS, MEASUREMENT_SCALE, FRAME, FRAME_EDGE, GRAY, BLUE_SOFT, SURFACE, CELL_LOW

CAMERA = {'eye': {'x': 0, 'y': -1.9, 'z': 1.65}, 'projection': {'type': 'perspective'}}
SCALE = MEASUREMENT_SCALE
FACES = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]


def box(x, y, z, w, d, h):
    return [(x,y,z),(x+w,y,z),(x+w,y+d,z),(x,y+d,z),
            (x,y,z+h),(x+w,y,z+h),(x+w,y+d,z+h),(x,y+d,z+h)]


def mesh(boxes, name):
    """One mesh per layer keeps 176 selectable cells inexpensive to render."""
    vertices, faces, colors, custom, labels = [], [], [], [], []
    for corners, color, key, label in boxes:
        offset = len(vertices)
        vertices.extend(corners)
        faces.extend(tuple(offset+i for i in face) for face in FACES)
        colors.extend([color]*8)
        custom.extend([[key]]*8)
        labels.extend([label]*8)
    return go.Mesh3d(x=[p[0] for p in vertices],y=[p[1] for p in vertices],z=[p[2] for p in vertices],
        i=[f[0] for f in faces],j=[f[1] for f in faces],k=[f[2] for f in faces],vertexcolor=colors,
        customdata=custom,text=labels,hovertemplate='%{text}<extra></extra>',name=name,showlegend=False,
        flatshading=True,lighting={'ambient':.75,'diffuse':.7,'specular':.12,'roughness':.8},
        lightposition={'x':-50,'y':-80,'z':100})


def outline(corners, label):
    x,y,z = [],[],[]
    for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        for v in (corners[a],corners[b],(None,None,None)):
            x.append(v[0]);y.append(v[1]);z.append(v[2])
    return go.Scatter3d(x=x,y=y,z=z,mode='lines',line={'color':BLUE,'width':7},hoverinfo='skip',name=label,showlegend=False)


def scene(fig, revision, height=440):
    axis={'visible':False,'showgrid':False,'zeroline':False,'showbackground':False}
    fig.update_layout(height=height,margin={'l':0,'r':0,'t':0,'b':0},paper_bgcolor='white',
        font={'family':'SUIT Variable, sans-serif','size':14,'color':TEXT,'weight':550},
        scene={'xaxis':axis,'yaxis':axis,'zaxis':axis,'camera':CAMERA,'aspectmode':'data',
               'dragmode':False,'bgcolor':'white','uirevision':revision},uirevision=revision,
        hoverlabel={'bgcolor':BLUE,'font':{'color':'white','size':14}},showlegend=False,dragmode=False)
    return fig


def quality_figure(data, mode='cell'):
    temperature=mode=='temperature'
    channels=data['snapshot']['temperatures' if temperature else 'cells']
    values=[p['value'] for p in channels if p['value'] is not None and math.isfinite(p['value'])]
    low,high=(min(values),max(values)) if values else (0,1)
    base,cells,centers=[],[],[]; selected=None
    unit='°C' if temperature else 'V'
    for module in range(1,17):
        x=((module-1)%4)*4.5; y=(3-(module-1)//4)*3.25
        base.append((box(x-.18,y-.18,-.2,4.32,2.86,.23),FRAME_EDGE,'',f'M{module:02d} · 논리 모듈'))
        # Neutral metal rails surround equal-sized, data-colored cells.
        for rx,ry,rw,rd in [(x-.18,y-.18,.13,2.86),(x+4.01,y-.18,.13,2.86),(x-.18,y-.18,4.32,.13),(x-.18,y+2.55,4.32,.13)]:
            base.append((box(rx,ry,-.1,rw,rd,.62),FRAME,'',f'M{module:02d}'))
        for bx,by in [(x-.15,y-.15),(x+3.94,y-.15),(x-.15,y+2.45),(x+3.94,y+2.45)]:
            base.append((box(bx,by,.52,.17,.17,.035),FRAME_EDGE,'',f'M{module:02d}'))
        centers.append((x+1.9,y-.42,.8,f'M{module:02d}'))
    for p in channels:
        module=int(p['id'][1:3]);channel=int(p['id'][-2:])
        x=((module-1)%4)*4.5; y=(3-(module-1)//4)*3.25
        corners=box(x+(channel-1)*(2.0 if temperature else .36),y,.04,1.8 if temperature else .30,2.4,.48)
        value=p['value']; valid=value is not None and math.isfinite(value)
        color=sample_colorscale(SCALE,[(value-low)/(high-low) if valid and high>low else .5])[0] if valid else GRAY
        # Red means a raw measurement check, never an invented cell defect diagnosis.
        if p['invalid']:color=RED
        if data.get('statusColors'):
            color=STATUS_COLORS[p.get('status', 'missing')]
        text=f"{p['id']}<br>{value:.3f} {unit}" if valid else f"{p['id']}<br>측정값 없음"
        text+=f"<br>원본값 확인 필요" if p['invalid'] else ''
        if data.get('statusColors'):
            text += '<br>'+{'normal':'정상','warning':'경고','danger':'위험','missing':'자료 없음'}[p.get('status','missing')]
            if p.get('z') is not None:
                text += f" · 상대 편차 {abs(p['z']):.2f}σ"
        cells.append((corners,color,p['id'],text))
        if p['id']==data['selectedCell']:selected=corners
    fig=go.Figure([mesh(base,'모듈 받침'),mesh(cells,'측정 채널')])
    fig.update_layout(scene_annotations=[dict(x=x,y=y,z=z,text=label,showarrow=False,font={'size':13,'color':BLUE},bgcolor=SURFACE,borderpad=3) for x,y,z,label in centers])
    # Thin top seams keep equal-valued adjacent cells individually legible.
    ex,ey,ez=[],[],[]
    for corners,_,_,_ in cells:
        for idx in (4,5,6,7,4):
            a,b,c=corners[idx];ex.append(a);ey.append(b);ez.append(c+.005)
        ex.append(None);ey.append(None);ez.append(None)
    fig.add_trace(go.Scatter3d(x=ex,y=ey,z=ez,mode='lines',line={'color':FRAME,'width':.8},hoverinfo='skip',showlegend=False))
    if selected:fig.add_trace(outline(selected,'선택 셀'))
    if values and not data.get('statusColors'):
        fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode='markers',hoverinfo='skip',
            marker={'color':[low],'colorscale':SCALE,'cmin':low,'cmax':high if high>low else low+1e-6,
                'showscale':True,'colorbar':{'title':unit,'thickness':10,'len':.6}},showlegend=False))
    return scene(fig,'quality-'+data['testId']+'-'+mode)


def maintenance_figure(data, points, selected=None):
    items=[]; chosen=None
    for p in points:
        index=int(p['page'])-1; x=(index%13)*1.15;y=(2-index//13)*1.6
        corners=box(x,y,0,.9,1.25,.34)
        color=TEXT if p['power']==0 else RED if p['prediction'] else CELL_LOW
        items.append((corners,color,str(p['row']),f"지점 {p['page']:02d} · {p['power']:,.1f} W<br>위험비 {p['risk']:.2f}배<br>{p['agreement']}"))
        if p['row']==selected:chosen=corners
    fig=go.Figure([mesh([(box(-.25,-.25,-.23,15.15,5.25,.2),FRAME,'','논리 지점판')],'받침'),mesh(items,'용접 지점')])
    fig.update_layout(scene_annotations=[dict(x=((int(p['page'])-1)%13)*1.15+.45,
        y=(2-(int(p['page'])-1)//13)*1.6+.625,z=.37,text=f"{p['page']:02d}",showarrow=False,
        font={'color':'white' if p['prediction'] or p['power']==0 else TEXT,'size':13},captureevents=False) for p in points])
    if chosen:fig.add_trace(outline(chosen,'선택 지점'))
    scene(fig,'maintenance-'+data['run']+'-'+str(points[0]['cycle']),330)
    fig.update_layout(scene_camera={'eye':{'x':0,'y':-1.8,'z':1.7},'projection':{'type':'perspective'}})
    return fig


def graph3d(figure, id):
    figure.update_layout(height=None,autosize=True)
    return html.Div(dcc.Graph(id=id,figure=figure,config={
        'displayModeBar':False,'responsive':True,'scrollZoom':False,'doubleClick':False},
        responsive=True, className='inspection-graph'),
        className='spatial-canvas fixed-inspection '+('weld-object' if id=='maintenance-3d' else 'battery-object'))


def selection(click, allowed):
    """Only accept data IDs actually present in the currently shown dataset."""
    points=(click or {}).get('points') or []
    custom=points[0].get('customdata') if points else None
    value=custom[0] if isinstance(custom,list) and custom else None
    return value if value in allowed else None


