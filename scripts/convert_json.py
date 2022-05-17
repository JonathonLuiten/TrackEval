
""" convert_json.py
Run example:
convert_json.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL CIWT
"""

from abc import ABC, abstractmethod
import argparse
from enum import Enum
import json
import os
import typing as ty


class DetectionEvent:
    def __init__(self,
                 *,
                 view_id: str,
                 frame_num: int,
                 detection_box_tlbr: ty.List[float],
                 object_id: ty.Optional[str] = None,
                 vehicle_class: ty.Optional[str] = None,
                 vehicle_confidence: ty.Optional[float] = None,
                 world_ground_point_xyz: ty.Optional[ty.List[float]] = None,
                 clipped: ty.Optional[bool] = None) -> None:
        self.view_id = view_id
        self.frame_num = frame_num
        self.detection_box_tlbr = detection_box_tlbr
        self.object_id = object_id
        self.vehicle_class = vehicle_class
        self.vehicle_confidence = vehicle_confidence
        self.world_ground_point_xyz = world_ground_point_xyz
        self.clipped = clipped

    @staticmethod
    def FromJSON(data, fps: float = 30):
        return DetectionEvent(
            view_id=data['viewId'],
            frame_num=int(float(data['timestampMs']) / 1000 * fps) + 1,  # add the 1 to avoid a 0 index frame_num
            object_id=data['objectId'] if 'objectId' in data else None,
            detection_box_tlbr=data['detectionBoxTLBR'],
            vehicle_class=data['vehicleClassConfidence']['vehicleClass'] if 'vehicleClassConfidence' in data else None,
            vehicle_confidence=data['vehicleClassConfidence']['confidence'] if 'vehicleClassConfidence' in data else None,
            world_ground_point_xyz=data['worldGroundPointXYZ'] if 'worldGroundPointXYZ' in data else None,
            clipped=data['clipped'] if 'clipped' in data else None)


class DetectionSequence:
    def __init__(self):
        self.name: str = None
        self.start_frame: int = None
        self.duration: int = None
        self.events: ty.List[DetectionEvent] = []

    def isEmpty(self) -> bool:
        return len(self.events) == 0

    def includesEventFrame(self, event: DetectionEvent) -> bool:
        if self.start_frame is None or self.duration is None:
            return False
        return self.start_frame >= event.frame_num and event.frame_num < self.start_frame + self.duration

    def isAdjacentTo(self, event: DetectionEvent) -> bool:
        return event.frame_num + 1 == self.start_frame or self.start_frame + self.duration == event.frame_num

    def addEvent(self, event: DetectionEvent):
        self.events.append(event)
        if self.start_frame is None or self.start_frame > event.frame_num:
            self.start_frame = event.frame_num
        if self.duration is None or self.start_frame + self.duration < event.frame_num:
            self.duration = event.frame_num - self.start_frame + 1

    def sortEvents(self):
        self.events.sort(key=lambda e: e.frame_num)


class Converter(ABC):
    def convertSequence(self, seq: DetectionSequence):
        for event in seq.events:
            event_str = converter.convert(event)
            if not event_str is None:
                print(event_str)

    @abstractmethod
    def convert(self, event: DetectionEvent) -> str:
        return NotImplemented


class MOTChallengeClass(Enum):
    PEDESTRIAN = 1
    PERSON_ON_VEHICLE = 2
    CAR = 3
    BICYCLE = 4
    MOTORBIKE = 5
    NON_MOT_VEHICLE = 6
    STATIC_PERSON = 7
    DISTRACTOR = 8
    OCCLUDER = 9
    OCCLUDER_ON_GROUND = 10
    OCCLUDER_FULL = 11
    REFLECTION = 12
    CROWD = 13


class MOTChallengeConverter(Converter):
    def __init__(self):
        self.object_counter = 0
        self.object_id_map = {}

    def convert(self, event: DetectionEvent) -> ty.Optional[str]:
        frame_num = event.frame_num

        if event.object_id is None:
            object_id = -1
        else:
            if not event.object_id in self.object_id_map:
                self.object_counter += 1
                self.object_id_map[event.object_id] = self.object_counter
            object_id = self.object_id_map[event.object_id]

        top = event.detection_box_tlbr[0]
        left = event.detection_box_tlbr[1]
        bottom = event.detection_box_tlbr[2]
        right = event.detection_box_tlbr[3]
        width = right - left
        height = bottom - top

        conf = MOTChallengeClass.PEDESTRIAN

        world_x = event.world_ground_point_xyz[0] if event.world_ground_point_xyz else -1
        world_y = event.world_ground_point_xyz[1] if event.world_ground_point_xyz else -1
        world_z = event.world_ground_point_xyz[2] if event.world_ground_point_xyz else -1

        return f'{frame_num},{object_id},{left:0.4f},{top:0.4f},{width:0.4f},{height:0.4f},{conf.value},{world_x:0.2f},{world_y:0.2f},{world_z:0.2f}'


class KITTIConverter(Converter):
    def __init__(self):
        self.object_counter = 0
        self.object_id_map = {}

    def convert(self, event: DetectionEvent) -> ty.Optional[str]:
        frame_num = event.frame_num

        if event.object_id is None:
            track_id = -1
        else:
            if not event.object_id in self.object_id_map:
                self.object_counter += 1
                self.object_id_map[event.object_id] = self.object_counter
            track_id = self.object_id_map[event.object_id]

        vehicle_type = event.vehicle_class if event.vehicle_class else 'DontCare'

        truncated = 1 if event.clipped else 0
        occluded = 3  # 3 for unknown

        alpha = 0  # not sure how to set this - unknown

        top = event.detection_box_tlbr[0]
        left = event.detection_box_tlbr[1]
        bottom = event.detection_box_tlbr[2]
        right = event.detection_box_tlbr[3]

        world_x = event.world_ground_point_xyz[0] if event.world_ground_point_xyz else -1
        world_y = event.world_ground_point_xyz[1] if event.world_ground_point_xyz else -1
        world_z = event.world_ground_point_xyz[2] if event.world_ground_point_xyz else -1

        rotation_y = 0  # not sure how to set this - unknown

        return f'{frame_num},{track_id},{vehicle_type},{truncated},{occluded},{alpha},{left},{top},{right},{bottom},{world_x},{world_y},{world_z},{rotation_y}'


format_type_converters = {
    'mot-challenge': MOTChallengeConverter(),
    'kitti': KITTIConverter(),
}
format_keys_str = ', '.join([id for id in format_type_converters])

def validate_file(str):
    if os.path.isfile(str):
        return str
    raise RuntimeError('not a valid path')


def validate_format(str):
    if str in format_type_converters:
        return str
    raise RuntimeError(f'not a valid format, please use one of the following\n{format_keys_str}.')


def convert_json_to_object_sequences(data, view_filter: ty.List[str] = []) -> ty.List[DetectionSequence]:
    obj_sequences: ty.Dict[str, DetectionSequence] = {}
    for event_json in data['events']:
        event = DetectionEvent.FromJSON(event_json)

        if len(view_filter) > 0 and not event.view_id in view_filter:
            continue

        if not event.object_id in obj_sequences:
            obj_sequences[event.object_id] = DetectionSequence()
            obj_sequences[event.object_id].name = event.object_id
        obj_sequences[event.object_id].addEvent(event)

    sequences: ty.List[DetectionSequence] = []
    for object_id in obj_sequences:
        obj_sequences[object_id].sortEvents()
        sequences.append(obj_sequences[object_id])
    sequences.sort(key=lambda e: e.start_frame)

    return sequences


def merge_adjacent_or_overlaping_sequences(sequences: ty.List[DetectionSequence]) -> ty.List[DetectionSequence]:
    merged_sequences: ty.List[DetectionSequence] = []
    for seq in sequences:
        if len(merged_sequences) == 0:
            merged_sequences.append(seq)
        elif merged_sequences[-1].includesEventFrame(seq.events[0]) or merged_sequences[-1].isAdjacentTo(seq.events[0]):
            for event in seq.events:
                merged_sequences[-1].addEvent(event)
        else:
            merged_sequences.append(seq)

    for seq in merged_sequences:
        seq.sortEvents()

    return merged_sequences


def fill_sequence_gaps_empty(sequence: DetectionSequence):
    frame_num = sequence.events[0].frame_num
    events: ty.List[DetectionEvent] = []
    for event in sequence.events:
        for i in range(frame_num, event.frame_num):
            empty_event_at_timestamp = DetectionEvent(
                view_id=event.view_id,
                frame_num=i,
                detection_box_tlbr=[-1, -1, -1, -1],
                object_id=None,
                vehicle_class=None,
                vehicle_confidence=0,
                world_ground_point_xyz=[-1, -1, -1],
                clipped=None)
            events.append(empty_event_at_timestamp)
        events.append(event)
        frame_num = event.frame_num + 1
    sequence.events = events


def fill_sequence_gaps_interp_0(sequence: DetectionSequence):
    frame_num = sequence.events[0].frame_num
    last_event: DetectionEvent = None
    events: ty.List[DetectionEvent] = []
    for event in sequence.events:
        for i in range(frame_num, event.frame_num):
            empty_event_at_timestamp = DetectionEvent(
                view_id=event.view_id,
                frame_num=i,
                detection_box_tlbr=last_event.detection_box_tlbr,
                object_id=last_event.object_id,
                vehicle_class=last_event.vehicle_class,
                vehicle_confidence=0,
                world_ground_point_xyz=last_event.world_ground_point_xyz,
                clipped=last_event.clipped)
            events.append(empty_event_at_timestamp)
        events.append(event)
        last_event = event
        frame_num = event.frame_num + 1
    sequence.events = events


def fill_sequence_gaps_interp_1(sequence: DetectionSequence):
    frame_num = sequence.events[0].frame_num
    last_event: DetectionEvent = None
    events: ty.List[DetectionEvent] = []
    for event in sequence.events:
        for i in range(frame_num, event.frame_num):
            alpha = (i - frame_num) / (event.frame_num - frame_num)
            empty_event_at_timestamp = DetectionEvent(
                view_id=event.view_id,
                frame_num=i,
                detection_box_tlbr=[
                    alpha * last_event.detection_box_tlbr[0] + (1 - alpha) * event.detection_box_tlbr[0],
                    alpha * last_event.detection_box_tlbr[1] + (1 - alpha) * event.detection_box_tlbr[1],
                    alpha * last_event.detection_box_tlbr[2] + (1 - alpha) * event.detection_box_tlbr[2],
                    alpha * last_event.detection_box_tlbr[3] + (1 - alpha) * event.detection_box_tlbr[3],
                ],
                object_id=last_event.object_id,
                vehicle_class=last_event.vehicle_class,
                vehicle_confidence=0,
                world_ground_point_xyz=[
                    alpha * last_event.world_ground_point_xyz[0] + (1 - alpha) * event.world_ground_point_xyz[0],
                    alpha * last_event.world_ground_point_xyz[1] + (1 - alpha) * event.world_ground_point_xyz[1],
                    alpha * last_event.world_ground_point_xyz[2] + (1 - alpha) * event.world_ground_point_xyz[2],
                ],
                clipped=last_event.clipped)
            events.append(empty_event_at_timestamp)
        events.append(event)
        last_event = event
        frame_num = event.frame_num + 1
    sequence.events = events


def rename_sequence_by_frame_num(seq: DetectionSequence):
    seq.name = f'{seq.start_frame:08}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=validate_file, help='Path to DetectionEventsCollection json file')
    parser.add_argument('--format', '-f', type=validate_format, default='mot-challenge', help=f'Output format type: {format_keys_str}')
    parser.add_argument('--view-filter', '-v', type=str, nargs='+', default=[], help=f'ViewIds to include in output')
    parser.add_argument('--interpolate', '-i', type=bool, default=False, help=f'Interpolates events, useful for ground truth')

    args = parser.parse_args().__dict__
    with open(args['path'], 'r') as f:
        data = json.load(f)

        obj_sequences = convert_json_to_object_sequences(data, view_filter=args['view_filter'])
        if args['interpolate']:
            for seq in obj_sequences:
                fill_sequence_gaps_interp_1(seq)

        sequences = merge_adjacent_or_overlaping_sequences(obj_sequences)

        converter: Converter = format_type_converters[args['format']]
        for seq in sequences:
            rename_sequence_by_frame_num(seq)
            converter.convertSequence(seq)