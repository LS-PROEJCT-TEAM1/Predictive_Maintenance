import unittest
from backend.data import Repository
from frontend.spatial import quality_figure, maintenance_figure, selection
from frontend.views import TABS


class SpatialInspectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo=Repository()

    def test_quality_mesh_preserves_all_channel_ids_and_values(self):
        data=self.repo.quality()
        figure=quality_figure(data)
        trace=figure.data[1]
        self.assertEqual(len(trace.x),176*8)
        self.assertEqual({r[0] for r in trace.customdata},{p['id'] for p in data['snapshot']['cells']})
        for i,p in enumerate(data['snapshot']['cells']):
            self.assertIn(f"{p['value']:.3f}",trace.text[i*8])
        # Geometry height never encodes voltage, preventing fake physical deformation.
        self.assertEqual(len({round(trace.z[i+4]-trace.z[i],4) for i in range(0,len(trace.z),8)}),1)

    def test_temperature_mode_preserves_32_channels(self):
        d=self.repo.quality()
        f=quality_figure(d,'temperature')
        self.assertEqual(len(f.data[1].customdata),32*8)

    def test_welding_points_link_to_source_rows(self):
        d=self.repo.maintenance()
        points=[p for p in d['points'] if p['cycle']==1]
        f=maintenance_figure(d,points)
        self.assertEqual(len(points),39)
        self.assertEqual({r[0] for r in f.data[1].customdata},{str(p['row']) for p in points})
        for i,p in enumerate(points):
            if p['power']==0:self.assertEqual(f.data[1].vertexcolor[i*8],'#172033')

    def test_selection_rejects_background_and_unrelated_id(self):
        allowed={'M02CV01'}
        self.assertEqual(selection({'points':[{'customdata':['M02CV01']}]},allowed),'M02CV01')
        for value in (None,{}, {'points':[]},{'points':[{'customdata':['']}]},{'points':[{'customdata':['M99CV99']}]}):
            self.assertIsNone(selection(value,allowed))

    def test_fixed_inspector_keeps_selected_welding_point(self):
        from frontend.maintenance_workspace import cycle_detail
        import json
        from plotly.utils import PlotlyJSONEncoder
        data=self.repo.maintenance()
        points=[p for p in data['points'] if p['cycle']==1]
        selected=str(points[12]['row'])
        def find(value):
            if isinstance(value,dict):
                if value.get('props',{}).get('id')=='m-point-select':return value['props']['value']
                for item in value.values():
                    result=find(item)
                    if result is not None:return result
            elif isinstance(value,list):
                for item in value:
                    result=find(item)
                    if result is not None:return result
        rendered=json.loads(json.dumps(cycle_detail(data,'1',selected),cls=PlotlyJSONEncoder))
        self.assertEqual(find(rendered),selected)

    def test_fixed_camera_keeps_clicks_without_view_controls(self):
        from frontend.spatial import graph3d
        quality=quality_figure(self.repo.quality())
        maintenance=self.repo.maintenance()
        weld=maintenance_figure(maintenance,[p for p in maintenance['points'] if p['cycle']==1])
        for figure in (quality,weld):
            self.assertFalse(figure.layout.scene.dragmode)
            self.assertEqual(figure.layout.scene.camera.eye.x,0)
            graph=graph3d(figure,'inspection').children
            self.assertFalse(graph.config['scrollZoom'])
            self.assertFalse(graph.config['doubleClick'])
            self.assertNotIn('staticPlot',graph.config)  # Click/hover selection stays active.

    def test_operational_navigation_has_no_presentation_tabs(self):
        self.assertEqual([t[0] for t in TABS['overview']],['summary'])
        for domain in ('demand','maintenance','quality'):
            self.assertEqual([t[0] for t in TABS[domain]],['analysis','review'])


if __name__=='__main__':unittest.main()
