import  torch;
import  torch.nn.functional  as F;
import  torch.nn             as nn;
import  torch.optim          as optim;
from    torch.utils.data     import Dataset;
from    torch.utils.data     import DataLoader;



class EERLoss(nn.Module):
    def __init__(self, device, beta, SETS_PER_BATCH, SAMPLES_PER_SET, binary_search_steps):
        super(EERLoss, self).__init__()

        self.device = device;
        self.beta = beta;
        self.SETS_PER_BATCH  = SETS_PER_BATCH;
        self.SAMPLES_PER_SET = SAMPLES_PER_SET;
        self.binary_search_steps = binary_search_steps;

        LEN = SAMPLES_PER_SET * SETS_PER_BATCH;

        legitimate_mask = torch.zeros((LEN, LEN), dtype=torch.float32).to(device);
        impostor_mask = torch.ones((LEN, LEN), dtype=torch.float32).to(device);
        impostor_mask = torch.tril(impostor_mask, diagonal=-1) 

        scalar_one = 1.0
        scalar_zero = 0.0
        vector_one = torch.tensor([scalar_one])
        vector_zero = torch.tensor([scalar_zero])

        for i in range(SETS_PER_BATCH):
            for j in range(SAMPLES_PER_SET):
                for k in range(j):
                    position = [SAMPLES_PER_SET * i + j, SAMPLES_PER_SET * i + k]
                    legitimate_mask[position[0], position[1]] = vector_one
                    impostor_mask[position[0], position[1]] = vector_zero    

        self.legitimate_mask = legitimate_mask;
        self.impostor_mask = impostor_mask;
    

    def get_units(self):
        return "";

    def get_name(self):
        return "scaled overlap area";


    def forward(self, input, target = None):
      embeddings = input;
      pairwise_distances = F.pairwise_distance(embeddings.unsqueeze(1), embeddings.unsqueeze(0), p=2)
      
      legitimate_scores = torch.masked_select(pairwise_distances, self.legitimate_mask.bool());
      impostor_scores = torch.masked_select(pairwise_distances, self.impostor_mask.bool());

      uL = torch.mean(legitimate_scores)
      uI = torch.mean(impostor_scores)
      
      lr = torch.stack([0.5 * uL, 1.5 * uI])
      for i in range(self.binary_search_steps):
          K = torch.tensor(100.0, dtype=torch.float32)
          d = torch.mean(lr)

          fn = torch.tanh(K * (torch.maximum(d, legitimate_scores) - d))
          frr = 100.0 * torch.mean(fn)

          fp = torch.tanh(K * (d - torch.minimum(d, impostor_scores)))
          far = 100.0 * torch.mean(fp)

          m = torch.maximum(frr, far)
          cl = torch.exp(K * (frr - m))
          cr = torch.exp(K * (far - m))

          nl = lr[0] * (1.0 - cl) + d * cl
          nr = lr[1] * (1.0 - cr) + d * cr

          lr = torch.stack([nl, nr])
      
      deer = torch.mean(lr)
      
      fn = torch.tanh(K * (torch.maximum(d, legitimate_scores) - d))
      frr = 100.0 * torch.mean(fn)

      fp = torch.tanh(K * (d - torch.minimum(d, impostor_scores)))
      far = 100.0 * torch.mean(fp)
      
      tl = torch.sum(torch.pow(torch.maximum(torch.tensor(1e-7), (legitimate_scores - deer) / deer), self.beta));
      tl /= legitimate_scores.shape[0];
      tl = 100.0 * torch.pow(tl, 1.0 / self.beta);
      tr = torch.sum(torch.pow(torch.maximum(torch.tensor(1e-7), (deer - impostor_scores) / deer), self.beta));
      tr /= impostor_scores.shape[0];
      tr = 100.0 * torch.pow(tr, 1.0 / self.beta);
      retval = tl + tr

      print(f"[eerArea-torch] deer={deer:.2f}    far={far:.2f}    frr={frr:.2f}    tl={tl:.4f}    tr={tr:.4f}    loss={retval:.6f}")
      return retval;

def get_loss(device, beta, SETS_PER_BATCH, SAMPLES_PER_SET, binary_search_steps):
    return EERLoss(device, beta, SETS_PER_BATCH, SAMPLES_PER_SET, binary_search_steps);


