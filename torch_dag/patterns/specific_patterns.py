from torch import nn

from torch_dag import structured_modules as smodules
from torch_dag.patterns.crude_patterns import VertexSpec

PRUNABLE_ATTENTION_PATTERN = {
    'named_specs':  {
        'final_matmul':     VertexSpec(
            module_class=smodules.TfMatmulModule,
            spec_dict={'transpose': False},
        ),
        'softmax':          VertexSpec(
            module_class=nn.Softmax,
        ),
        'linear':           VertexSpec(
            module_class=nn.Linear,
        ),
        'attention_matmul': VertexSpec(
            module_class=smodules.TfMatmulModule,
            spec_dict={'transpose': True},
        ),
    },
    'starting_key': 'final_matmul',
    'pattern':      {
        'final_matmul':     ['softmax', 'linear'],
        'softmax':          ['attention_matmul', ],
        'attention_matmul': ['linear', 'linear'],
    }
}
