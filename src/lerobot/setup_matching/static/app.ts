type BBox = { x_min: number; y_min: number; x_max: number; y_max: number };
type RoiObject = { id: string; label: string; bbox: BBox };
type FramePayload = { width: number; height: number; image: string };

declare const state: {
  referenceFrame: FramePayload | null;
  streamFrame: FramePayload | null;
  objects: RoiObject[];
};

export function centroid(bbox: BBox): [number, number] {
  return [(bbox.x_min + bbox.x_max) / 2, (bbox.y_min + bbox.y_max) / 2];
}

export function bboxFromPoints(start: { x: number; y: number }, current: { x: number; y: number }): BBox {
  const xMin = Math.min(start.x, current.x);
  const yMin = Math.min(start.y, current.y);
  const xMax = Math.max(start.x, current.x);
  const yMax = Math.max(start.y, current.y);
  return {
    x_min: clamp(xMin, 0, 0.99),
    y_min: clamp(yMin, 0, 0.99),
    x_max: clamp(Math.max(xMax, xMin + 0.01), 0.01, 1),
    y_max: clamp(Math.max(yMax, yMin + 0.01), 0.01, 1),
  };
}

export function matchPayload(params: {
  referenceDatasetPath: string;
  replayDatasetPath: string;
  cameraKey: string;
  referenceEpisodeIndex: number;
  referenceFrameIndex: number;
  replayEpisodeIndex: number;
  streamFrames: number;
  save: boolean;
  objects: RoiObject[];
}) {
  return {
    reference_dataset_path: params.referenceDatasetPath,
    replay_dataset_path: params.replayDatasetPath,
    camera_key: params.cameraKey,
    reference_episode_index: params.referenceEpisodeIndex,
    reference_frame_index: params.referenceFrameIndex,
    replay_episode_index: params.replayEpisodeIndex,
    replay_start_frame_index: 0,
    stream_frames: params.streamFrames,
    save: params.save,
    objects: params.objects.map((object) => ({ label: object.label, bbox: object.bbox })),
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
