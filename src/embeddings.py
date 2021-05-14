
from torch import nn

def _get_subnet(parent_net, break_layer_idx = '1'):
    """
    Get subnet from given network at index
    """
    children_list = []
    for n,c in parent_net.named_children():
        children_list.append(c)
        if n == break_layer_idx:
            break
    sub_net = nn.Sequential(*children_list)
    return sub_net

def get_embeddings(batch, network, layers):
    subnets = []
    for layer_id in layers:
        subnets.append(_get_subnet(network, layer_id))

    embeddings = []
    for subnet in subnets:
        embeddings.append(subnet(batch).detach().numpy())

    return embeddings