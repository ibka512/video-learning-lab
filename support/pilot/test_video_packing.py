"""Verify the real Qwen temporal processor retains single-frame signals and source times."""
import unittest
import numpy as np
from PIL import Image
from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLVideoProcessor
from vision import prepare_temporal_inputs


class Tokenizer:
    def __call__(self,prompt,**kwargs):
        return {"input_ids":np.array([[1,2,3]]),"attention_mask":np.array([[1,1,1]])}


class Processor:
    def __init__(self):
        self.video_processor=Qwen3VLVideoProcessor()
        self.tokenizer=Tokenizer()
    def apply_chat_template(self,messages,**kwargs):
        return "<|vision_start|><|video_pad|><|vision_end|>"+messages[0]["content"][1]["text"]


def unpack(pixels,grid,processor):
    gt,gh,gw=map(int,grid)
    ps=processor.patch_size;ms=processor.merge_size;tps=processor.temporal_patch_size
    original=(1,gt,tps,3,gh//ms,ms,ps,gw//ms,ms,ps)
    perm=(0,1,4,7,5,8,3,2,6,9)
    after=tuple(original[i] for i in perm)
    return np.asarray(pixels).reshape(after).transpose(tuple(np.argsort(perm))).reshape(gt*tps,3,gh*ps,gw*ps)


class TemporalPackingContracts(unittest.TestCase):
    def test_every_unique_frame_and_odd_tail_survives_real_preprocessor(self):
        count=9
        colors=[np.array([i*23,255-i*19,i*13],dtype=np.uint8) for i in range(count)]
        images=[Image.fromarray(np.broadcast_to(c,(64,96,3)).copy()) for c in colors]
        evidence=[{"frame_id":f"F{i}","time":i*0.07} for i in range(count)]
        p=Processor()
        inputs,prompt,receipt=prepare_temporal_inputs(p,"inspect",images,evidence,258048)
        restored=unpack(inputs["pixel_values_videos"],receipt["video_grid_thw"][0],p.video_processor)
        self.assertEqual(len(restored),10)
        self.assertEqual(receipt["padding_frames"],1)
        for i in range(10):
            wanted=(colors[min(i,8)].astype(float)/255-0.5)/0.5
            np.testing.assert_allclose(restored[i].mean(axis=(1,2)),wanted,atol=0.002)
        ids=[x for g in receipt["temporal_groups"] for x in g["frame_ids"]]
        self.assertEqual(ids,[f"F{i}" for i in range(count)]+["F8"])

    def test_one_frame_flash_is_preserved_in_temporal_pixel_payload(self):
        arrays=[np.zeros((64,96,3),dtype=np.uint8) for _ in range(8)]
        arrays[5][:]=[255,0,0]
        images=[Image.fromarray(x) for x in arrays]
        p=Processor()
        inputs,_,receipt=prepare_temporal_inputs(p,"inspect",images,
            [{"frame_id":f"F{i}","time":i/30} for i in range(8)],258048)
        frames=unpack(inputs["pixel_values_videos"],receipt["video_grid_thw"][0],p.video_processor)
        self.assertGreater(frames[5,0].mean(),0.99)
        self.assertLess(frames[4,0].mean(),-0.99)
        self.assertLess(frames[6,0].mean(),-0.99)

    def test_video_timestamp_labels_use_actual_source_times(self):
        p=Processor()
        images=[Image.new("RGB",(96,64)) for _ in range(4)]
        _,prompt,_=prepare_temporal_inputs(p,"inspect",images,
            [{"frame_id":f"F{i}","time":t} for i,t in enumerate([0,0.1,0.4,0.6])],258048)
        self.assertIn("<0.050 seconds>",prompt)
        self.assertIn("<0.500 seconds>",prompt)


if __name__=="__main__":
    unittest.main()
