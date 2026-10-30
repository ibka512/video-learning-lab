"""Contract tests for missing-frame detection, checkpoints and transcript preservation."""
import json
from pathlib import Path
import tempfile
import unittest
from transcribe import restore_transcript, timestamp
from vision import windows, parse_observation, all_successes
from audit import check_frame_coverage


class CoverageContracts(unittest.TestCase):
    def test_windows_cover_tail_and_never_skip_any_source_frame(self):
        for n in [1,2,7,8,9,14,15,2065]:
            spans=list(windows(n))
            covered={i for a,b in spans for i in range(a,b)}
            self.assertEqual(covered,set(range(n)))
            self.assertEqual(spans[-1][1],n)
            self.assertLessEqual(max(b-a for a,b in spans),8)

    def test_one_missing_frame_prevents_complete_status(self):
        source=[{"frame_id":f"F{i}","part_time":i/30} for i in range(3)]
        request={"status":"success","window_index":0,"frame_ids":["F0","F1"],
                 "source_inputs":[{"frame_id":"F0","time":0},{"frame_id":"F1","time":1/30}],
                 "preprocessor":{"image_grid_thw":[[1,2,2],[1,2,2]],
                                 "pixel_values_shape":[8,1536],"image_pad_token_count":2,
                                 "expected_image_tokens":2},"observation":{"visible":"state"}}
        result=check_frame_coverage(source,[request,request])
        self.assertEqual(result["successful_unique_frames"],2)
        self.assertEqual(result["missing_frame_ids"],["F2"])
        self.assertFalse(result["strict_input_complete"])

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):list(windows(100,8,8))

    def test_evidence_cannot_reference_unsubmitted_frame(self):
        text=json.dumps({"visible":"按钮变化","changes":[{"frame_id":"F9999999","text":"点击"}],"uncertain":[]})
        with self.assertRaises(ValueError):parse_observation(text,{"F0000000"})

    def test_schema_example_echo_is_not_a_visual_observation(self):
        text=json.dumps({"visible":"窗口中可确认的简短画面描述","changes":[],"uncertain":[]})
        with self.assertRaises(ValueError):parse_observation(text,{"F0"})

    def test_unlocalized_change_keeps_window_scope_without_inventing_frame(self):
        text=json.dumps({"visible":"窗口发生变化","changes":["背景出现碎裂"],"uncertain":[]})
        result=parse_observation(text,{"F0","F1"})
        self.assertIsNone(result["changes"][0]["frame_id"])
        self.assertEqual(result["changes"][0]["localization"],"window_only")
        self.assertEqual(result["changes"][0]["source_frame_ids"],["F0","F1"])

    def test_non_object_model_response_is_rejected(self):
        with self.assertRaises(ValueError):parse_observation("[]",{"F0"})

    def test_changed_model_revision_does_not_reuse_cached_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/"request-000000.json"
            p.write_text(json.dumps({"identity":{"revision":"old"},"status":"success","window_index":0}))
            with self.assertRaises(RuntimeError):all_successes(Path(temporary),{"revision":"new"})



class TranscriptContracts(unittest.TestCase):
    def test_restore_punctuation_and_preserve_raw_transcript(self):
        text="设置 Godot，值为 １.２！"
        tokens=["设","置","Godot","值","为","1","2"]
        units=[{"word_id":f"W{i}","text":v,"start":i,"end":i+1} for i,v in enumerate(tokens)]
        result=restore_transcript(text,units)
        self.assertEqual("".join(x["text"] for x in result),text)
        self.assertIn("１.",result[-2]["text"])

    def test_omitted_critical_number_cannot_be_silently_aligned(self):
        with self.assertRaises(ValueError):
            restore_transcript("设置值为42",[
                {"text":"设","start":0,"end":1},{"text":"置","start":1,"end":2},
                {"text":"值","start":2,"end":3},{"text":"为","start":3,"end":4},
                {"text":"4","start":4,"end":5}
            ])

    def test_timestamp_rollover_is_valid(self):
        self.assertEqual(timestamp(59.9996),"00:01:00.000")


if __name__=="__main__":
    unittest.main()
