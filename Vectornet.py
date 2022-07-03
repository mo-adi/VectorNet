import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from GNNs import MLP, MLP2, Global_Graph, Sub_Graph
from utils import (batch_init, get_dis_point_2_points, merge_tensors,
                   to_origin_coordinate)


class VectorNet(nn.Module):
    def __init__(self, hidden_size, device):
        super(VectorNet, self).__init__()

        self.sub_graph = Sub_Graph(hidden_size)
        self.global_graph = Global_Graph(hidden_size)
        self.predict_traj = MLP2(hidden_size, 6*30*2 + 6)
        self.device = device
        self.hidden_size = hidden_size
        self.traj_completion_criterion = nn.SmoothL1Loss()

    def forward_encode_sub_graph(self, mapping, matrix, polyline_spans, batch_size):
        # len(input_list_list) = batch_size
        # len(input_list_list[i]) = #polylines in scene i
        # input_list_list[i][j] = #vectors in polyline j of scene i
        input_list_list = []
        for i in range(batch_size):
            input_list = []
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(
                    matrix[i][polyline_span], device=self.device)
                input_list.append(tensor)

            input_list_list.append(input_list)

        element_states_batch = []

        # Aim is getting a feature for each polyline
        #   from many vectors of it
        for batch_idx in range(batch_size):
            hidden_states, lengths = merge_tensors(
                input_list_list[batch_idx], self.device, self.hidden_size)
            # input_list_list[i] -> (8, 64), (6, 64), (19, 64)
            # hidden_states -> (3, 19, 64)

            hidden_states = self.sub_graph(hidden_states, lengths)
            # hidden_states.shape = (#polylines, hidden_size)
            element_states_batch.append(hidden_states)

        return element_states_batch

    def forward(self, mapping, validate=False):

        matrix = [i["matrix"] for i in mapping]
        polyline_spans = [i["polyline_spans"] for i in mapping]
        labels = [i["labels"] for i in mapping]

        batch_init(mapping) # len(mapping): 16 -> list of dictionaries
        batch_size = len(matrix)

        element_states_batch = self.forward_encode_sub_graph(
            mapping, matrix, polyline_spans, batch_size)

        inputs, inputs_lengths = merge_tensors(
            element_states_batch, self.device, self.hidden_size)
        # inputs.shape = (batch_size, max(#polylines), hidden_size)

        max_poly_num = max(inputs_lengths) # 73
        attention_mask = torch.zeros(
            [batch_size, max_poly_num, max_poly_num], device=self.device) # shape: [16, 73, 73]
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask=attention_mask, mapping=mapping)

        outputs = self.predict_traj(hidden_states[:, 0, :])
        pred_probs = F.log_softmax(outputs[:, -6:], dim=-1)
        outputs = outputs[:, :- 6].view([batch_size, 6, 30, 2]) # shape: [16, 6, 30, 2]
        
        for i in range(batch_size):
            for j in outputs[i]:
                for pred in j:
                    argmin_pred = np.array(pred.tolist()).argmin()
            gt_points = np.array(labels[i]).reshape([30, 2]) # [30, 2] 
            gt_tensor = torch.from_numpy(gt_points).to(device='cuda')
            closest_pred = outputs[i][argmin_pred].to(dtype=torch.float64) # find prediction with closest endpoint to gt
            loss_func = nn.MSELoss()
            loss = loss_func(closest_pred, gt_tensor)  # find loss over predicted trajectory argmin

        if validate:
            outputs = np.array(outputs.tolist())
            pred_probs = np.array(pred_probs.tolist(
            ), dtype=np.float32) if pred_probs is not None else pred_probs
            for i in range(batch_size):
                for each in outputs[i]:
                    to_origin_coordinate(each, i)
            return outputs, pred_probs

        return loss