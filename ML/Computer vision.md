# DETR: Paper explained

Basic: CNN + Transformer
![[Pasted image 20240306131722.png]]
Transformer for DETR looks like this: 
![[Pasted image 20240306131858.png]]



## Архитектура DETR

DETR состоит из 3-х частей:

- **CNN-бэкбон** для **формирования feature map **;
- **Encoder-Decoder трансформер**;
- **Feed-Forward Network (FFN)** для формирования финального предсказания детекции.

### CNN-бэкбон

Авторы использовали модели семейства ResNet, а в качестве baseline backbone — **ResNet-50**, как уже было сказано выше, формирует карту признаков исходного изображения.
На входе принимается изображение размером $x \in \mathbb{R}^{3 * H_{0} * W_{0}}$, а на выходе — тензор размером $2048 * H * W; H = \frac{H_{0}}{32}, W = \frac{W_{0}}{32}$.
### Transformer part 1: Encoder

Далее мы уменьшаем канальную размерность в feature map с помощью 1D свертки ($2048 → d, d < 2048$) и разворачиваем ее в 1D размерность, сворачивая пространственную размерность $H * W$.  

Конвертация в 1D вектор нужна потому, что энкодер принимает на вход последовательность. Поскольку **MHSA (MultiHead Self-Attention)** энкодера инвариантен к перестановкам во входной последовательности — необходимо добавить **фиксированное позиционное кодирование (spatial positional encoding)**, которое поможет сети учитывать порядок фич в карте признаков.

> **Важный момент:** как и в ViT, энкодер представляет собой набор $N$ последовательных блоков, состоящих из Multi-Head Self-Attention и FFN, где размерность входной последовательности совпадает с размером выходной.

> **Еще один важный момент:** в self-attention $Query == Key$, а позиционное кодирование добавляется только к ним, ведь мы хотим дать информацию от других объектов только content-части. Также, в отличие от классического трансформера и ViT, мы добавляем позиционное кодирование в каждый слой. Это же верно и для декодера!

#### EncoderLayer code

```
import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
		"""
		Имплементация слоя энкодера трансформера.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Слои для FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
				
				# Функция для выбора активации из списка: [ReLU, GELU, GLU]
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # Forward для случая, когда нормализация идёт после MHSA и MLP.
        # Добавляем позиционное кодирование только к Query и Key.
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # Forward для случая, когда нормализация идёт до MHSA и MLP.
        src2 = self.norm1(src)
        # Добавляем позиционное кодирование только к Query и Key.
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
```
#### Encoder code
```
import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
		"""
		Имплементация энкодера трансформера.
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
				
				# Стакаем N раз энкодер слои.
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
```
### Transformer part 2: Decoder

Decoder в DETR устроен так же, как и в [оригинальной работе](https://arxiv.org/pdf/1706.03762.pdf) 2017 года. Однако есть небольшое отличие: DETR использует параллельный декодинг вместо авторегрессионного.

- **авторегрессионный декодинг**: целевое предложение генерируется последовательно токен за токеном, отправляя частичный результат в качестве входных данных для следующей итерации авторегрессии, вплоть до длины $m$ целевого предложения.
- 
- **параллельный декодинг**: этот метод изменяет только алгоритм декодирования и может использоваться поверх любой модели авторегрессии без изменений. Алгоритмы параллельного декодирования обрабатывают все предложение или блок из $b$ токенов параллельно: исходные токены (PAD токены) постепенно уточняются с помощью $k$ шагов, пока не будет достигнуто условие остановки. **Важно отметить:** $k \leq m$ гарантирует качество и общее ускорение декодирования.

В декодере также применяется **позиционное кодирование**, причем целых два:
1. **Spatial positional encoding** — такое же кодирование, как и в энкодере. Как мы уже говорили выше, авторы хотели полностью отказаться от prior knowledge, который несет в себе классические анкоры. По сути их роль на себя и взяло позиционное кодирование. При этом у него нет явного геометрического смысла, и он полностью учится с нуля на данных в обучении. Таким образом, мы дали дополнительную информацию детектору о взаимном расположении токенов и не внесли prior knowledge, поскольку positional encoding — обучаемый параметр.
2. **Object queries** постепенно формируются в ходе декодинга и отвечают за сбор визуальной информации о текущем объекте интереса с помощью скрытого состояния энкодера. Перед первым слоем инициализируются как нулевые векторы.

> **Важный момент: максимальное количество детекций равно числу object queries.**

**Стоит отметить:** в каждом слое декодера используются два вида механизма внимания:

1. **Self-attention** служит обмену информацией между object queries. Как и в энкодере, для V они не добавляются.
2. **Cross-attention**. В этой части object queries смотрят на результат работы энкодера и поглощают визуальную информацию. В качестве $Q$ в данном случае выступает сумма object queries с самой собой после MHSA, а вот $K$ и $V$ здесь другие — это выход энкодера с positional embedding и без него соответственно. Таким образом, каждый object query производит некий [SoftPooling](https://arxiv.org/pdf/2101.00440v3.pdf) релевантных визуальных фичей из тех или иных частей изображения. В какой-то степени этот модуль заменяет традиционный RoIPooling, только объекты могут считывать информацию со всего изображения, а не только из ограниченной области.


#### DecoderLayer
```
import torch
import torch.nn as nn


class TransformerDecoderLayer(nn.Module):
		"""
		Имплементация слоя декодера трансформера.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Слои для FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
				
				# Функция для выбора активации из списка: [ReLU, GELU, GLU]
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # Forward для случая, когда нормализация идёт после MHSA и MLP.
        # Добавляем позиционное кодирование только к Query и Key.
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Обратите внимание на то, что подаётся как Q, K, V в attention.
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        # Forward для случая, когда нормализация идёт до MHSA и MLP.
        tgt2 = self.norm1(tgt)
        # Добавляем позиционное кодирование только к Query и Key.
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        # Обратите внимание на то, что подаётся как Q, K, V в attention.
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
```
#### Decoder
```
import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
		"""
		Имплементация декодера трансформера.
    """
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
				
				# Стакаем M раз декодер слои и записываем их в список intermediate.
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)
```




### Transformer: Finale
```
import torch
import torch.nn as nn


class Transformer(nn.Module):
		"""
		Имплементация трансформера.
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
				
				# Инициализируем энкодер.
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
				
				# Инициализируем декодер.
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # Разворачиваем входной тензор размера NxCxHxW в тензор размера HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
```
### FFN

FFN состоит из двух независимых частей:

1. **MLP (MultiLayer Perceptron) блок** — последовательность из 3-х линейных слоев с функцией активации ReLU. Предсказывает нормализованные значения центра бокса, его высоты и ширины.
2. **Линейный слой**, предсказывающий класс бокса с помощью функции softmax.

Поскольку в конце мы предсказываем $N$ (фиксированное число; обычно гораздо больше, чем количество искомых объектов на изображении) bbox-ов, нужно добавить специальный класс “no object” — ∅. Он играет такую же роль, что и класс “background” в обычных CNN детекторах.
```
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
		Имплементация MLP.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        # output dim = 4, так как bbox = [x, y, w, h]
        # num_layers = 3
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # если input_dim = hidden_dim = 512, то размерности 3-х Linear слоёв - 
	      # (512, 512), (512, 512), (512, 4)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
```
### DETR
```
import torch
import torch.nn as nn


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
	  """
	  Вспомогательная функция.
	  """
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # оптимизация для ONNX
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class DETR(nn.Module):
    """
		Имплементация DETR.
    """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """
        Инициализация модели.
        Parameters:
            backbone: CNN бэкбон.
            transformer: Encoder-Decoder transformer.
            num_queries: количество object queries. это максимальное число
	            детекций DETR на одно изображение. Для COCO авторы
	            рекомендуют брать число 100.
            
            aux_loss: True если используется вспомогательный лосс для декодинга.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # num_classes + 1, потому что мы не забываем про no_objects
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """
        На вход forward принимает вложенный тензор, который состоит из:
        The forward expects a NestedTensor, which consists of:
             - samples.tensor: батч картинок размера [batch_size x 3 x H x W].
             - samples.mask: бинарная маска размера [batch_size x H x W],
             содержащая 1 на padded пикселях.
						
			Forward возвращает dict со следующими элементами:
             - "pred_logits": классификационные логиты, включая no-object,
             для всех queries.
              Размер = [batch_size x num_queries x (num_classes + 1)]
              
             - "pred_boxes": нормализованные координаты боксов 
             значения от 0 до 1) для всех queries размера
              (center_x, center_y, height, width).
              
             - "aux_outputs": если aux_loss == True, возвращает их значения.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
				
				# классификационная голова
        outputs_class = self.class_embed(hs)
        # регрессионная голова
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

```
## Loss
Осталось только как-то соотнести их с ground truth боксами и метками. На первый взгляд может показаться, что это несложно. К сожалению, не все так просто: порядок предсказаний не совпадает с порядком ground truth.

Как же их тогда можно сматчить? Использовать IoU для поиска ближайших боксов? Но такой матчинг точно не будет всегда оптимальным. И здесь к нам на помощь приходит **комбинаторная оптимизация** — она обеспечит нахождение лучшего one-to-one матчинга, который, в свою очередь, даст минимально возможный суммарный лосс. Для этого используется **“Венгерский алгоритм” (Hungarian algorithm)**, работающий за полиномиальное $O(n^{3})$ время. Ниже о нем будет рассказано подробнее. Также стоит отметить, что в функции [linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) из scipy используется не сам венгерский алгоритм, а его более быстрая [модификация](https://ieeexplore.ieee.org/document/7738348).

Вернемся к лоссу. Он, как и в обычных детекторах, складывается из суммы лоссов классификации и локализации. В данном случае используются кросс-энтропия, L1 и [](https://giou.stanford.edu/)[Generalized IoU](https://giou.stanford.edu/). Количество предсказаний (равное количеству object queries) почти всегда будет больше, чем количество реальных ground truth объектов на картинке, поэтому “лишние” предсказания отправляются в класс “no object” и не передаются в итоговой набор боксов.
### Венский алгоритм
Начнем с общей формулировки “задачи о назначениях”, которую он решает:
```
Имеется некоторое число работ и некоторое число исполнителей. Любой исполнитель может быть назначен на выполнение любой (но только одной) работы, но с неодинаковыми затратами. Нужно распределить работы так, чтобы выполнить работы с минимальными затратами.
```
Если же вернуться к проблеме матчинга для лосса в DETR, то нам нужно придумать, как посчитать матрицу стоимости. Все очень просто: она состоит из взвешенной суммы (по дефолту все веса = 1) трех перечисленных выше лоссов.

## Downstream tasks

Для задачи **panoptic** (мы еще хотим разделять объекты одного и того же класса) сегментации авторы добавили сегментационную голову к DETR, так же, как Faster R-CNN был расширен до Mask R-CNN. Давайте разберем это более подробно.

- Для начала получаем боксы, как и раньше;
- Далее мы получаем attention map из MHSA;
- Мы хотим маски, поэтому накидываем сегментационную голову, которая будет предсказывать бинарные маски. При этом мы объединяем feature map (с бэкбона) c масками боксов, полученных на предыдущем этапе. Размеры итогового тензора будут зависеть от количества боксов;
- Для определения финального предсказания мы формируем [FPN-like](https://arxiv.org/pdf/1612.03144.pdf) архитектуру из полученных выше feature map.

> **Важный момент:** **Feature Pyramid Net (FPN),** или **пирамида признаков** — свёрточная нейронная сеть, построенная в виде пирамиды и служащая для объединения достоинств карт признаков нижних и верхних уровней сети; первые имеют высокое разрешение, но низкую семантическую, обобщающую способность; вторые наоборот.

### Код MHAttentionMap, отвечающего только за подсчет attention softmax (без умножения на V)
```
import torch
import torch.nn as nn


class MHAttentionMap(nn.Module):
    """
		Имплементация MHAttentionMap.
    """
    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
				
				# Инициализация весов.
        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights
```

### Код MaskHeadSmallConv — FPN-like CNN головы
```
import torch
import torch.nn as nn


class MaskHeadSmallConv(nn.Module):
		"""
		Имплементация MaskHeadSmallConv. Upsampling делается с помощью FPN подхода.
    """
    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()
				
				# В качестве нормализации активно используем GroupNorm.
				# Идея в том, чтобы построить feature map в разных масштабах: 
				# от большего к меньшему, которые затем мы будем складывать друг
				# с другом.
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
				
				# Инициализация весов.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
		    """
		    fpns - 3 feature map разного размера с бэкбона, которые ниже
		    мы будем склеивать с feature maps, полученными свёртками в mask head.
		    """
		    
		    # Мы объединяем x (feature map с backbone) и масками боксов,
		    # полученных на этапе MHAttentionMap. Размеры итогового тензора
		    # будут зависеть от количества боксов.
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)
				
				# Строим пирамиду из признаков.
        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
		        # expand - resize feature map cur_fpn до размеров feature map x.
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x
```


### Код SegmentationDetr
```
import torch
import torch.nn as nn


class DETRsegm(nn.Module):
		"""
		Имплементация сегментационного DETR.
    """
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr
				
				# Как было сказано выше, мы можем просто заморозить веса детектора.
				# Это не повлияет на итоговое качество, но существенно упростит 
				# обучение.
        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.detr.backbone(samples)

        bs = features[-1].tensors.shape[0]

        src, mask = features[-1].decompose()
        assert mask is not None
        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])

        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        # MHSA формирует attention maps для боксов.
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
				
				# А mask head, которая и учит маски объектов.
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        out["pred_masks"] = outputs_seg_masks
        return out


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)
```
## Достоинства DETR

- Взаимодействие object queries через self-attention декодера вместе с использованием matching loss теоретически приводят к **отсутствию дубликатов предсказаний.** Однако на практике дубликаты предсказаний все-таки встречаются, поэтому накинуть NMS стоит.
- Как и в ViT, self-attention отлично справляется с задачей **учета глобального контекста** и моделированием отношений между далекими друг от друга токенами (патчами из изображения), превосходя обычные CNN детекторы.
- Слой [MultiHeadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) выполняет похожий трюк в CV, что и в NLP: каждая голова обучается независимо от других и берет на себя (или вместе с другой частью голов) какую-либо подзадачу. Например, в NLP такие задачи могут брать на себя головы:
    - **позиционная**: как токены расположены друг относительно друга; что идет до / после;
    - **синтаксическая**: отслеживание некоторых основных синтаксических отношений в предложении (подлежащее-глагол, глагол-объект);
    - **частотность токенов**: отслеживание наименее частых токенов.
  В случае детекции можно выделить подзадачи локализации и классификации:
	- для предсказания координат нужны границы объекта;
	- для классификации — фокусировка на семантически важных частях.

## Недостатки DETR

- **Плохое качество на маленьких объектах**. DETR использует только один scale из бэкбона, который имеет слишком маленькое разрешение для точной детекции небольших объектов. Почему при этом нельзя добавить FPN и использовать разрешение повыше / всю пирамиду фичей? Ответ простой и грустный: операция self-attention в энкодере и cross-attention в декодере очень чувствительны к размерности фичей, потому что attention имеет квадратичную зависимость от них 😞.
- **Проблемы обучения**. Для достижения адекватных метрик DETR-у, как и многим трансформерам, нужно на порядок больше эпох, чем аналогичным классическим детекторам. Ну и в целом на практике очень тяжело обучать большие трансформерные архитектуры: нужно много данных, критически важен learning rate и scheduling, чтобы лосс не улетел в NaN и так далее.
- **Проблемы инференса**. Проблемы здесь из-за того, что DETR — трансформер. Это означает большие затраты по времени и памяти на инференсе вследствие квадратичной сложности Attention (мы должны посчитать скор попарно между всеми токенами). К тому же, не будем забывать, что Query, Key и Value — обучаемые параметры, которые вносят свой вклад в увеличение latency через MAC.

# CLIP: Contrastive Language-Image Pre-Training  

**Premise** 
CLIP, which stands for Contrastive Language-Image Pretraining, is a deep learning model developed by OpenAI in 2021. CLIP’s embeddings for images and text share the same space, enabling direct comparisons between the two modalities. This is accomplished by training the model to bring related images and texts closer together while pushing unrelated ones apart.

## CLIP : Idea

CLIP is designed to predict which N × N potential (image, text) pairings within the batch are actual matches. To achieve this, CLIP establishes a multi-modal embedding space through the joint training of an image encoder and text encoder. **The CLIP loss aims to maximize the cosine similarity between the image and text embeddings for the N genuine pairs in the batch while minimizing the cosine similarity for the N² − N incorrect pairings.** The optimization process involves using a symmetric cross-entropy loss function that operates on these similarity scores. The following presents pseudocode (taken from the original paper) outlining the core implementation of CLIP.

```
# image_encoder - ResNet or Vision Transformer  
# text_encoder - CBOW or Text Transformer  
# I[n, h, w, c] - minibatch of aligned images  
# T[n, l] - minibatch of aligned texts  
# W_i[d_i, d_e] - learned proj of image to embed  
# W_t[d_t, d_e] - learned proj of text to embed  
# t - learned temperature parameter  

# extract feature representations of each modality  
I_f = image_encoder(I) #[n, d_i]  
T_f = text_encoder(T) #[n, d_t]  

# joint multimodal embedding [n, d_e]  
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)  
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)  

# scaled pairwise cosine similarities [n, n]  
logits = np.dot(I_e, T_e.T) * np.exp(t)  

# symmetric loss function  
labels = np.arange(n)  
loss_i = cross_entropy_loss(logits, labels, axis=0)  
loss_t = cross_entropy_loss(logits, labels, axis=1)  
loss = (loss_i + loss_t)/2# image_encoder - ResNet or Vision Transformer  

# text_encoder - CBOW or Text Transformer  
# I[n, h, w, c] - minibatch of aligned images  
# T[n, l] - minibatch of aligned texts  
# W_i[d_i, d_e] - learned proj of image to embed  
# W_t[d_t, d_e] - learned proj of text to embed  
# t - learned temperature parameter  
# extract feature representations of each modality  

I_f = image_encoder(I) #[n, d_i]  
T_f = text_encoder(T) #[n, d_t]  

# joint multimodal embedding [n, d_e]  
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)  
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)  

# scaled pairwise cosine similarities [n, n]  
logits = np.dot(I_e, T_e.T) * np.exp(t)  

# symmetric loss function  
labels = np.arange(n)  
loss_i = cross_entropy_loss(logits, labels, axis=0)  
loss_t = cross_entropy_loss(logits, labels, axis=1)  
loss = (loss_i + loss_t)/2
```

## Model architecture
ClIP uses two separate architectures as the backbone for encoding vision and text datasets:
1. `image_encoder`: Represents the neural network architecture (e.g., ResNet or Vision Transformer) responsible for encoding images.
2. `text_encoder`: Represents the neural network architecture (e.g., CBOW, BERT, or Text Transformer) responsible for encoding textual information.
![[Pasted image 20240417234537.png]]

The model takes a batch of n pairs of images and texts as input where:
- `I[n, h, w, c]`: Represents a minibatch of aligned images, where `n` is the batch size, `h` is the image height, `w` is the image width, and `c` is the number of channels.
- `T[n, l]`: Represents a minibatch of aligned texts, where `n` is the batch size, and `l` is the length of the textual sequence.

### Projections

- `W_i[d_i, d_e]`: Represents the learned projection matrix for mapping image features (`I_f`) to an embedding space (`I_e`). The shape of `W_i` is `[d_i, d_e]`, where `d_e` is the desired dimensionality of the joint embedding space.
- `W_t[d_t, d_e]`: Represents the learned projection matrix for mapping text features (`T_f`) to the same embedding space (`T_e`). The shape of `W_t` is `[d_t, d_e]`.

```
class Projection(nn.Module):  
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:  
        super().__init__()  
        self.linear1 = nn.Linear(d_in, d_out, bias=False)  
        self.linear2 = nn.Linear(d_out, d_out, bias=False)  
        self.layer_norm = nn.LayerNorm(d_out)  
        self.drop = nn.Dropout(p)  
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        embed1 = self.linear1(x)  
        embed2 = self.drop(self.linear2(F.gelu(embed1)))  
        embeds = self.layer_norm(embed1 + embed2)  
        return embeds
```

### Embedding and Normalization
- `I_e = l2_normalize(np.dot(I_f, W_i), axis=1)`: Embeds and normalizes image features in the joint embedding space (`I_e`).
- `T_e = l2_normalize(np.dot(T_f, W_t), axis=1)`: Embeds and normalizes text features in the joint embedding space (`T_e`).

```
class VisionEncoder(nn.Module):  
    def __init__(self, d_out: int) -> None:  
        super().__init__()  
        base = models.resnet34(pretrained=True)  
        d_in = base.fc.in_features  
        base.fc = nn.Identity()  
        self.base = base  
        self.projection = Projection(d_in, d_out)  
        for p in self.base.parameters():  
            p.requires_grad = False  
  
    def forward(self, x):  
        projected_vec = self.projection(self.base(x))  
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)  
        return projected_vec / projection_len  
  
  
class TextEncoder(nn.Module):  
    def __init__(self, d_out: int) -> None:  
        super().__init__()  
        self.base = AutoModel.from_pretrained(Config.text_model)  
        self.projection = Projection(Config.transformer_embed_dim, d_out)  
        for p in self.base.parameters():  
            p.requires_grad = False  
  
    def forward(self, x):  
        out = self.base(x)[0]  
        out = out[:, 0, :]  # get CLS token output  
        projected_vec = self.projection(out)  
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)  
        return projected_vec / projection_len  
  
vision_encoder = VisionEncoder(Config.embed_dim)  
I_e = vision_encoder(images)  
caption_encoder = TextEncoder(Config.embed_dim)          
T_e = caption_encoder(text["input_ids"])
```

`logits = np.dot(I_e, T_e.T) * np.exp(t)`: Computes pairwise cosine similarities between image and text embeddings, scaled by a learned temperature parameter `t`.

### Symmetric Loss Function

CLIP uses contrastive loss (first introduced in [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)) to bring related images and texts closer together while pushing unrelated ones apart.

```
def CLIP_loss(logits: torch.Tensor) -> torch.Tensor:  
    n = logits.shape[1]      # number of samples  
    labels = torch.arange(n) # Create labels tensor  

    loss_i = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")  
    loss_t = F.cross_entropy(logits, labels, reduction="mean")  

    loss = (loss_i + loss_t) / 2  
  
    return loss
```

### Custom simple model
```
class CustomModel(nn.Module):  
    def __init__(self, lr: float = 1e-3) -> None:  
        super().__init__()  
        self.vision_encoder = VisionEncoder(Config.embed_dim)  
        self.caption_encoder = TextEncoder(Config.embed_dim)  
        self.tokenizer = Tokenizer(AutoTokenizer.from_pretrained(Config.text_model))  
        self.lr = lr  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
  
    def forward(self, images, text):  
        text = self.tokenizer(text).to(self.device)  
  
        image_embed = self.vision_encoder(images)  
        caption_embed = self.caption_encoder(text["input_ids"])  
        similarity = caption_embed @ image_embed.T  
  
        loss = CLIP_loss(similarity)  
        img_acc, cap_acc = metrics(similarity)  
        return loss, img_acc, cap_acc
```
## Custom dataset training
### Dataset
```
class Flickr30kDataset(torch.utils.data.Dataset):  
    def __init__(self):  
        self.dataset = load_dataset("nlphuji/flickr30k", cache_dir="./huggingface_data")  
        self.transform = transforms.Compose([  
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
        ])  
        self.cap_per_image = 2  
  
    def __len__(self):  
        return self.dataset.num_rows["test"] * self.cap_per_image  
  
    def __getitem__(self, idx):  
        original_idx = idx // self.cap_per_image  
        image = self.dataset["test"][original_idx]["image"].convert("RGB")  
        image = self.transform(image)  
  
        # labels  
        caption = self.dataset["test"][original_idx]["caption"][idx % self.cap_per_image]  
  
        return {"image": image, "caption": caption}  
  
flickr30k_custom_dataset = Flickr30kDataset()

from dataclasses import dataclass  
  
  
@dataclass  
class Config:    
    embed_dim: int = 512  # Embedding dimension  
    transformer_embed_dim: int = 768  # Transformer embedding dimension  
    max_len: int = 32  # Maximum text length  
    text_model: str = "distilbert-base-multilingual-cased"  # Text model name  
    epochs: int = 3 # Number of training epochs  
    batch_size: int = 128 # Batch size
```
Model: 
```
# Create an instance of your model  
model = CustomModel().to(device)  
  
# Define optimizer  
optimizer = torch.optim.Adam([  
    {'params': model.vision_encoder.parameters()},  
    {'params': model.caption_encoder.parameters()}  
], lr=model.lr)
```
Training:
```
batch_zero = True  
for epoch in range(start_epoch, num_epochs):  
    model.train()  
    for batch in clip_dataloader:  
        image = batch["image"].to(device)  
        text = batch["caption"]  
        # images, text = batch  
        loss, img_acc, cap_acc = model.common_step((image, text))  
  
        # Backward pass and optimization  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
  
        if batch_zero:  
          print(f"Epoch [{0}/{num_epochs}], Batch Loss: {loss.item()}")  
          batch_zero = False  
  
  
    # Print training statistics  
    print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item()}")  
  
print("Training complete.")
```
## Interacting with API


# DINO: Self-Supervised Vision Transformers
![[Pasted image 20240418002650.png]]**A Student ViT learns to predict global features in an image from local patches supervised by the cross entropy loss from a momentum Teacher ViT’s embeddings while doing centering and sharpening to prevent mode collapse**

**_Pseudocode_**:
![[Pasted image 20240418003631.png]]

## Premise

The network learns through a process called ‘==self-distillation==’. There is a teacher and student network both having the same architecture, a **Vision Transformer(ViT)**.

The teacher is a **momentum teacher** which means that it’s weights are an exponentially weighted average of the student’s. The momentum teacher was introduced in the paper “_Momentum Contrast for Unsupervised Visual Representation Learning”_ in order to prevent mode collapse when the teacher and the student are the same and output the same embeddings regardless of the input.

The update rule for the teacher’s weights are:
![[Pasted image 20240418002950.png]]with λ following a **cosine schedule from 0.996 to 1** during training in this paper.

As is common in self-supervised learning, different crops of one image are taken. Small crops are called Local views( <50% of the image) and large crops( >50% of the image) are called Global views.

All crops are passed through the student while only the global views are passed through the teacher. **_This encourages “local-to-global” correspondence, training the student to interpolate context from a small crop._**

Random augmentations of color jittering, Gaussian blur and solarization are also applied on the views to make the network more robust.

## Loss
The teacher and student each predict a 1-dimensional embedding. A softmax along with cross entropy loss is applied to make student’s distribution match the teacher’s
**_So we are asking our student network to have the same proportions of features as the teacher._** **_The teacher having a larger context predicts more high level features which the student must also match._**

The cross-entropy loss tries to make the two distributions the same just as in knowledge distillation.

This can also be seen as a made up classification problem. **_We are asking our network to make up a classification problem such that the network can learn meaningful global representations from local views._**

## Centering and sharpening

There are two forms of mode collapse: regardless of the input, the model output is always the same along all the dimensions(i.e same output for any input) or dominated by one dimension. Centering and Sharpening aim to prevent both these.

**Centering**: The teacher’s raw activations have the their exponentially moving average subtracted from them. It simply is: 
```
Logits = Logits - Logits_mean
```
**_This means that activations must be sometimes positive when they are above their mean and sometimes negative when they are below._** This prevents any one feature from dominating as the mean will be somewhere in the middle of the range. And we know that softmax gives very low values to negative numbers and high values to positive ones.

**Sharpening_: Sharpening is the same as applying a temperature to the softmax to artificially make the distribution more peaked_**, i.e exaggerate small differences so that there is one or some high values and some low values. This prevents all the activations from being the same value as small differences are exaggerated. This acts in synergy with centering which keeps changing which activations are high. Sharpening also helps the student get a stronger signal which features it should increase.
