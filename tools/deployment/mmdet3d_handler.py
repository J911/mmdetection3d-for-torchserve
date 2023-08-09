# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.structures.points import get_points_type
from pypcd import pypcd
from io import StringIO as sio

class MMdet3dHandler(BaseHandler):
    """MMDetection3D Handler used in TorchServe.

    Handler to load models in MMDetection3D, and it will process data to get
    predicted results. For now, it only supports SECOND.
    """
    threshold = 0.5
    load_dim = 4
    use_dim = [0, 1, 2, 3]
    coord_type = 'LIDAR'
    attribute_dims = None

    def initialize(self, context):
        """Initialize function loads the model in MMDetection3D.

        Args:
            context (context): It is a JSON Object containing information
                pertaining to the model artifacts parameters.
        """
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')
        self.model = init_model(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data):
        """Preprocess function converts data into LiDARPoints class.

        Args:
            data (List): Input data from the request.

        Returns:
            `LiDARPoints` : The preprocess function returns the input
                point cloud data as LiDARPoints class.
        """
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            pts = row.get('data') or row.get('body')
            if isinstance(pts, str):
                pts = base64.b64decode(pts)

            """ New scope for PCD file format <@j911>"""
            if row.get('format') is not None:
                file_format = str(row.get('format').decode('utf-8'))
                if file_format == "pcd":
                    fileobj = sio(str(pts.decode('utf-8')))
                    pcd_data = pypcd.point_cloud_from_fileobj(fileobj)
                    pts = np.zeros([pcd_data.width, 4], dtype=np.float32)
                    pts[:, 0] = pcd_data.pc_data['x'].copy()
                    pts[:, 1] = pcd_data.pc_data['y'].copy()
                    pts[:, 2] = pcd_data.pc_data['z'].copy()
                    pts[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32) / 255.
                else:
                    pts = np.frombuffer(pts, dtype=np.float32)
                    pts = pts.reshape(-1, self.load_dim)
                    pts = pts[:, self.use_dim]

            """ Ignore this code <@j911>"""
            # points = np.frombuffer(pts, dtype=np.float32)
            # points = points.reshape(-1, self.load_dim)
            # points = points[:, self.use_dim]
            # points_class = get_points_type(self.coord_type)
            # points = points_class(
            #     points,
            #     points_dim=points.shape[-1],
            #     attribute_dims=self.attribute_dims)

        return pts

    def inference(self, data):
        """Inference Function.

        This function is used to make a prediction call on the
        given input request.

        Args:
            data (`LiDARPoints`): LiDARPoints class passed to make
                the inference request.

        Returns:
            List(dict) : The predicted result is returned in this function.
        """
        results, _ = inference_detector(self.model, data)
        return results

    def postprocess(self, data):
        """Postprocess function.

        This function makes use of the output from the inference and
        converts it into a torchserve supported response output.

        Args:
            data (List[dict]): The data received from the prediction
                output of the model.

        Returns:
            List: The post process function returns a list of the predicted
                output.
        """
        scores = data.pred_instances_3d.scores_3d.detach().cpu().numpy().tolist()
        boxes = data.pred_instances_3d.bboxes_3d.detach().cpu().numpy().tolist()
        labels = data.pred_instances_3d.labels_3d.detach().cpu().numpy().tolist()
        output = [[]]
        for score, box, label in zip(scores, boxes, labels):
            anno = {
                    "obj_id": "",
                    "obj_type": "%d (%f)" % (label, score),
                    "psr": {
                        "position": {
                        "x": box[0],
                        "y": box[1],
                        "z": box[2] + box[5]/2
                        },
                        "rotation": {
                            "x": 0,
                            "y": 0,
                            "z": box[6]
                        },
                        "scale": {
                            "x": box[3],
                            "y": box[4],
                            "z": box[5]
                        }
                    }
                }
            output[0].append(anno)

        '''
        for pts_index, result in enumerate(data):
            output.append([])
            if 'pts_bbox' in result.keys():
                pred_bboxes = result['pts_bbox']['boxes_3d'].numpy()
                pred_scores = result['pts_bbox']['scores_3d'].numpy()
            else:
                pred_bboxes = result['boxes_3d'].numpy()
                pred_scores = result['scores_3d'].numpy()

            index = pred_scores > self.threshold
            bbox_coords = pred_bboxes[index].tolist()
            score = pred_scores[index].tolist()

            output[pts_index].append({'3dbbox': bbox_coords, 'score': score})
        '''
        return output
