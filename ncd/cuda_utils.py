import pycuda.autoinit
import pycuda.driver
import pycuda.compiler

class Holder(pycuda.driver.PointerHolderBase):
    """ This class is from
    https://gist.github.com/szagoruyko/440c561f7fce5f1b20e6154d801e6033
    https://discuss.pytorch.org/t/how-can-i-get-access-to-the-raw-gpu-data-of-a-tensor-for-pycuda-and-how-do-i-convert-back/21394/5
    """
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()

def get_max_set_size(len_states, num_levels=4):
    max_size, cur_len = 1, 0
    for i in range(len_states.bit_length()):
        max_size += int(num_levels ** i)
        cur_len += int(i * (num_levels ** i))
        if cur_len >= len_states:
            return max_size
    return max_size

def get_LZW_NCD(x, y, num_levels=4):
    assert x.size()[1:] == y.size()[1:]

    device_attrs = pycuda.driver.Context.get_device().get_attributes()
    max_dx = device_attrs[pycuda.driver.device_attribute.MAX_BLOCK_DIM_X]
    max_dy = device_attrs[pycuda.driver.device_attribute.MAX_BLOCK_DIM_Y]
    max_dz = device_attrs[pycuda.driver.device_attribute.MAX_BLOCK_DIM_Z]
    max_threads = device_attrs[pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK]

    max_gdx = device_attrs[pycuda.driver.device_attribute.MAX_GRID_DIM_X]
    max_gdy = device_attrs[pycuda.driver.device_attribute.MAX_GRID_DIM_Y]
    max_gdz = device_attrs[pycuda.driver.device_attribute.MAX_GRID_DIM_Z]

    _, num_filters, len_states = x.size()
    
    dx = int(min(2 ** (x.size(0).bit_length() - 1), max_dx))
    dy = int(min(2 ** (y.size(0).bit_length() - 1), max_dy, max_threads // dx))
    dz = int(min(num_filters, max_dz, int(max_threads // (dx * dy))))

    bdim = (dx, dy, dz)
    bnum = (x.size(0), y.size(0), num_filters)
    gdim = tuple([w // dw + (w % dw > 0) for w, dw in zip(bnum, bdim)])
    if gdim[0] > max_gdx or gdim[1] > max_gdy or gdim[2] > max_gdz:
        raise ValueError("Check the grid dimension")
    
    max_set_size = get_max_set_size(int(2 * len_states), num_levels)
    
    mod = pycuda.compiler.SourceModule("""
    __device__ void get_LZW_complexity(int *x, int *y, int& c_x, int& c_xy)
    {
        int LZWset[%(MAX_SET_SIZE)s * %(NUM_LEVELS)s] = {0, };
        int n = 0;
        int cur = 0;

        c_x = 1; c_xy = 1;
        for(int i = 0; i < %(LEN_STATES)s; ++i)
        {
            if(LZWset[cur + x[i]] == 0)
            {
                LZWset[cur + x[i]] = %(NUM_LEVELS)s * (++n);
                cur = 0;
                ++c_x ; ++c_xy;
            }
            else
                cur = LZWset[cur + x[i]];
        }

        for(int i = 0; i < %(LEN_STATES)s; ++i)
        {
            if(LZWset[cur + y[i]] == 0)
            {
                LZWset[cur + y[i]] = %(NUM_LEVELS)s * (++n);
                cur = 0;
                ++c_xy;
            }
            else
                cur = LZWset[cur + y[i]];
        }
    }

    __global__ void LZW_distances(int *x, int *y, float *ret)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int idy = threadIdx.y + blockDim.y * blockIdx.y;
        int idz = threadIdx.z + blockDim.z * blockIdx.z;
        int c_x, c_y, c_xy, c_yx;

        if(idx >= %(NUM_A)s || idy >= %(NUM_B)s || idz >= %(NUM_FILTERS)s)
            return;

        get_LZW_complexity(&x[(idx * %(NUM_FILTERS)s + idz) * %(LEN_STATES)s],
                          &y[(idy * %(NUM_FILTERS)s + idz) * %(LEN_STATES)s],
                          c_x, c_xy);
        get_LZW_complexity(&y[(idy * %(NUM_FILTERS)s + idz) * %(LEN_STATES)s],
                          &x[(idx * %(NUM_FILTERS)s + idz) * %(LEN_STATES)s],
                          c_y, c_yx);
        float ncd = (float) (min(c_xy, c_yx) - min(c_x, c_y)) / (float) max(c_x, c_y);
        ret[(idx * %(NUM_B)s + idy) * %(NUM_FILTERS)s + idz] = ncd;
    }

    """ % {
        'NUM_A' : x.size(0),
        'NUM_B' : y.size(0),
        'NUM_FILTERS' : num_filters,
        'LEN_STATES' : len_states,
        'MAX_SET_SIZE' : max_set_size,
        'NUM_LEVELS' : num_levels
    })
    func = mod.get_function('LZW_distances')
    
    return lambda x,y,z : func(Holder(x), Holder(y), Holder(z), block=bdim, grid=gdim)