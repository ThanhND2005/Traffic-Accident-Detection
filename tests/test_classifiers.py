import pytest
import numpy as np
from src.classifiers.rule_based import RuleBasedAccidentDetector
from src.postprocessing.post_processor import PostProcessor

def test_multi_vehicle_clustering():
    detector = RuleBasedAccidentDetector()
    
    # 3 vehicles in collision: A with B, B with C
    pair_candidates = [
        {
            "frame_id": 100,
            "tracks": [1, 2],
            "score": 0.75,
            "rules_triggered": 2,
            "reasons": "decel_a(10->2) + iou(0.2)",
            "bbox": [10.0, 10.0, 50.0, 50.0],
        },
        {
            "frame_id": 100,
            "tracks": [2, 3],
            "score": 0.80,
            "rules_triggered": 2,
            "reasons": "spin_b(60°) + iou(0.25)",
            "bbox": [30.0, 30.0, 80.0, 80.0],
        }
    ]
    
    current_items = [
        {"track_id": 1, "bbox": [10.0, 10.0, 40.0, 40.0]},
        {"track_id": 2, "bbox": [30.0, 30.0, 60.0, 60.0]},
        {"track_id": 3, "bbox": [50.0, 50.0, 80.0, 80.0]},
        {"track_id": 4, "bbox": [200.0, 200.0, 250.0, 250.0]}, # separate vehicle
    ]
    
    clusters = detector.cluster_multi_vehicle_candidates(pair_candidates, current_items, current_frame=100)
    assert len(clusters) == 1
    c = clusters[0]
    assert c["num_vehicles"] == 3
    assert set(c["tracks"]) == {1, 2, 3}
    assert 4 not in c["tracks"]
    assert c["score"] >= 0.80
    assert c["bbox"] == [10.0, 10.0, 80.0, 80.0]

def test_post_processor_multi_vehicle_extension():
    pp = PostProcessor(temporal_window=3, min_consecutive_alerts=2, fps=30.0)
    
    # Frame 1: pair (1, 2)
    c1 = {
        "frame_id": 10,
        "tracks": [1, 2],
        "score": 0.75,
        "bbox": [10.0, 10.0, 50.0, 50.0],
        "reasons": "decel + iou",
    }
    # Frame 2: pair (1, 2)
    c2 = {
        "frame_id": 11,
        "tracks": [1, 2],
        "score": 0.85,
        "bbox": [10.0, 10.0, 50.0, 50.0],
        "reasons": "decel + iou",
    }
    pp.process_frame_candidate(10, c1)
    conf = pp.process_frame_candidate(11, c2)
    assert conf is not None
    assert set(conf["tracks"]) == {1, 2}
    
    # Frame 15: vehicle 3 crashes into them
    c3 = {
        "frame_id": 15,
        "tracks": [2, 3],
        "score": 0.90,
        "bbox": [20.0, 20.0, 80.0, 80.0],
        "reasons": "multi_vehicle_crash",
    }
    pp.process_frame_candidate(15, c3)
    
    # Check that confirmed event was extended to include vehicle 3
    last_event = pp.get_summary()[-1]
    assert set(last_event["tracks"]) == {1, 2, 3}
    assert last_event["num_vehicles"] == 3
    assert last_event["frame_end"] == 15
