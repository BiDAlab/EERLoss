import  torch;
import  torch.nn.functional  as F;
import  torch.nn             as nn;
import  torch.optim          as optim;
from    torch.utils.data     import Dataset;
from    torch.utils.data     import DataLoader;



class EERLoss(nn.Module):
    def __init__(self, device, SETS_PER_BATCH, SAMPLES_PER_SET, binary_search_steps):
        super(EERLoss, self).__init__()

        self.device = device;
        self.SETS_PER_BATCH  = SETS_PER_BATCH;
        self.SAMPLES_PER_SET = SAMPLES_PER_SET;
        # el número de pasos en un proceso de búsqueda binaria que se usa para optimizar el cálculo del EER.
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
        return "%";

    def get_name(self):
        return "EER";


    def forward(self, input, target = None):
      embeddings = input;
      # Calcula la distancia entre todos los embeddings
      pairwise_distances = F.pairwise_distance(embeddings.unsqueeze(1), embeddings.unsqueeze(0), p=2)
      # almacenan las distancias entre pares legítimos e impostores, respectivamente, usando las máscaras definidas.
      legitimate_scores = torch.masked_select(pairwise_distances, self.legitimate_mask.bool());
      impostor_scores = torch.masked_select(pairwise_distances, self.impostor_mask.bool());
      # promedios de las distancias legítimas e impostoras.
      uL = torch.mean(legitimate_scores)
      uI = torch.mean(impostor_scores)
      
      lr = torch.stack([0.5 * uL, 1.5 * uI])
      # tanh -> torch.tanh se encuentra en el rango (−1,1)(−1,1) y transforma los valores de entrada en una escala que suaviza las diferencias.
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
      # el d final, umbral para tomar la ddecision
      deer = torch.mean(lr)
      
      fn = torch.tanh(K * (torch.maximum(d, legitimate_scores) - d))
      frr = 100.0 * torch.mean(fn)

      fp = torch.tanh(K * (d - torch.minimum(d, impostor_scores)))
      far = 100.0 * torch.mean(fp)

      retval = (far + frr) / 2.0;
      print(f"[eer-torch] deer={deer:.2f}    far={far:.2f}    frr={frr:.2f}    loss={retval:.6f}")
      return retval;


def get_loss(device, SETS_PER_BATCH, SAMPLES_PER_SET, binary_search_steps):
    return EERLoss(device, SETS_PER_BATCH, SAMPLES_PER_SET, binary_search_steps);

