import unittest
from ocr_routing import reasons,propose,from_native,select_anchors
from build_refined_evidence import overlap_line
from refinement_context import for_interval
class RoutingTests(unittest.TestCase):
 def test_high_confidence_key_still_routes(self):self.assertIn('configuration_or_path',reasons('display/window/per_pixel_transparency/allowed',1))
 def test_no_plain_caption_trigger(self):self.assertEqual(reasons('大家好，欢迎来到中子脉冲',1),[])
 def test_bad_ocr_key_routes(self):self.assertIn('configuration_or_path',reasons('display/window/per_pixeLtransparency/allowed',.9))
 def test_numeric_units_and_help(self):
  self.assertIn('typed_value_or_numeric_parameter',reasons('延迟 30ms',.99));self.assertTrue(reasons('注意：这个参数将在重启后生效',1))
 def test_merge_and_boundary(self):
  ts=[{'text':'a/b/c','confidence':1,'bbox':[1,2,80,15]},{'text':'bool=false','confidence':1,'bbox':[5,25,90,40]}]
  rs=propose(ts,100,60);self.assertEqual(len(rs),1);self.assertEqual(rs[0]['bbox'][0],0);self.assertLessEqual(rs[0]['bbox'][2],100);self.assertEqual(set(rs[0]['trigger_indices']),{0,1})
 def test_compound_identifier_without_extension(self):
  self.assertIn('compound_identifier',reasons('MeshRenderer',1))
  self.assertNotIn('compound_identifier',reasons('Welcome',1))
 def test_neighbor_line_is_whole_and_bounded(self):
  ts=[{'text':'保持比例','confidence':1,'bbox':[100,100,180,115]},
      {'text':'普通菜单项','confidence':1,'bbox':[100,79,180,94]},
      {'text':'远处说明','confidence':1,'bbox':[100,20,180,35]}]
  region=propose(ts,400,300)[0]['bbox'];self.assertLessEqual(region[1],67);self.assertGreater(region[1],35)
 def test_native_coordinates(self):
  row={'width':200,'height':100,'texts':[{'text':'abc','confidence':1,'bbox_bottom_left':[.1,.2,.3,.1]}]};t=from_native(row)[0];self.assertAlmostEqual(t['bbox'][1],70);self.assertAlmostEqual(t['bbox'][3],80)
 def test_line_alignment_not_other_row(self):self.assertFalse(overlap_line([0,0,100,10],[0,25,100,35]))
 def test_exact_frame_temporal_scope(self):
  r={'ocr_id':'PR1','frame_id':'F1','time':2.,'text':'a','apple_readings':[],'relation':'new_detection','bbox_top_left':[0,0,1,1],'basis':'onscreen_text','review_status':'unverified'}
  self.assertEqual(for_interval([r],1,2),[]);self.assertEqual(len(for_interval([r],2,3)),1)
if __name__=='__main__':unittest.main()
