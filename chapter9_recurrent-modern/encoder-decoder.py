from torch import nn

#@save
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        """
        定义编码器的前向传播。
        它接受一个长度可变的序列X，并将其转换为一个编码后的状态。
        这个方法必须在子类中被重写。
        """
        raise NotImplementedError

#@save
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        """
        初始化解码器的状态。
        这个方法的核心作用是将编码器的输出（enc_outputs）转换为解码器需要的初始状态。
        例如，在RNN中，这通常是编码器最后一个时间步的隐状态。
        这个方法必须在子类中被重写。
        """
        raise NotImplementedError

    def forward(self, X, state):
        """
        定义解码器的前向传播。
        它接受输入序列X（例如，目标语言序列的前n-1个词元）和初始状态，
        生成预测结果。
        这个方法必须在子类中被重写。
        """
        raise NotImplementedError

#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """
        定义整个编码器-解码器模型的前向传播逻辑。
        enc_X: 编码器的输入（源语言序列）
        dec_X: 解码器的输入（目标语言序列）
        """
        # 1. 将源序列输入编码器，得到编码后的输出
        enc_outputs = self.encoder(enc_X, *args)
        # 2. 用编码器的输出初始化解码器的状态
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # 3. 将解码器输入和初始状态送入解码器，得到最终输出
        return self.decoder(dec_X, dec_state)