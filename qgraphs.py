import networkx as nx
import math

class BinaryGraph(nx.Graph):
    """
    A Graph class that adds binary node representation capability.
    Can be used with any graph structure.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_binary_node_representation(self, start_node=0):
        """
        Returns a dictionary with nodes as keys and their binary representation as values.
        The number of bits is determined by the number of nodes in the graph.
        
        Parameters:
        start_node (int): The starting number for binary representation (default 0)
                         Set to 1 if your node numbering starts at 1
        """
        num_nodes = len(self.nodes())
        if num_nodes == 0:
            return {}
        
        # Calculate required number of bits: ceil(log2(num_nodes))
        num_bits = max(1, math.ceil(math.log2(num_nodes)))
        
        binary_rep = {}
        for node in self.nodes():
            # Adjust for start_node if nodes don't begin at 0
            adjusted_node = node - start_node
            # Format node number as binary with leading zeros
            binary_str = format(adjusted_node, f'0{num_bits}b')
            binary_rep[node] = binary_str
        
        return binary_rep