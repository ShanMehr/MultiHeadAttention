class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # d_model = 500
        # d_k=100
        # num_heads=5
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # You must extract and use the weights that are randomly generated in the following data structures
        torch.manual_seed(0)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_attention(self,Q,K,V,list_output=True):
        transposed_K = []
        for batch in range(len(K)):
            matrix=[]
            for seq in range(len(K[0])):
                matrix.append(self.transpose(K[batch][seq]))
            transposed_K.append(matrix)
        attn_scores=[]
        for batch in range(len(Q)):
            matrix=[]
            for seq in range(len(Q[0])):
                matrix.append(self.mat_mul(Q[batch][seq],transposed_K[batch][seq]))
            attn_scores.append(matrix)
        print(torch.tensor(attn_scores).size())
        for batch in range(len(attn_scores)):
            for seq in range(len(attn_scores[0])):
                for head in range(len(attn_scores[0][0])):
                    for d_model in range(len(attn_scores[0][0][0])):
                        attn_scores[batch][seq][head][d_model]/=math.sqrt(self.d_k)
        softmax_tensor = []
        for batch in range(len(attn_scores)):
            sequence = []
            for seq in range(len(attn_scores[0])):
                matrix=[]
                for head in range(len(attn_scores[0][0])):
                    head_probs = attn_scores[batch][seq][head]
                    exp_probs = [math.exp(prob) for prob in head_probs]
                    sum_exp_probs = sum(exp_probs)
                    softmax_probs = [exp_prob / sum_exp_probs for exp_prob in exp_probs]
                    matrix.append(softmax_probs)
                sequence.append(matrix)
            softmax_tensor.append(sequence)
        output=[]
        for batch in range(len(softmax_tensor)):
            matrix=[]
            for seq in range(len(softmax_tensor[0])):
                matrix.append(self.mat_mul(softmax_tensor[batch][seq], V[batch][seq]))
            output.append(matrix)
        if list_output is False:
            output=torch.tensor(output)
        return output
        
        
        
    def scaled_dot_product_attention(self,Q, K, V, list_output=True):
        transposed_K = []
        for batch in range(len(K)):
            transposed_K.append(self.transpose(K[batch]))
        print("Transposed",torch.tensor(transposed_K).size())
        attn_scores=[]
        for batch in range(len(Q)):
            print("1st matrix multiplication")
            attn_scores.append(self.mat_mul(Q[batch],transposed_K[batch]))
        for batch in range(len(attn_scores)):
            for seq in range(len(attn_scores[0])):
                for head in range(len(attn_scores[0][0])):
                    attn_scores[batch][seq][head]/=math.sqrt(self.d_k)
        softmax_tensor = []
        for batch in range(len(attn_scores)):
            sequence = []
            for seq in range(len(attn_scores[0])):
                head_probs = attn_scores[batch][seq]
                exp_probs = [math.exp(prob) for prob in head_probs]
                sum_exp_probs = sum(exp_probs)
                softmax_probs = [exp_prob / sum_exp_probs for exp_prob in exp_probs]
                sequence.append(softmax_probs)
            softmax_tensor.append(sequence)
                    
        softmax_tensor = []
        for batch in range(len(attn_scores)):
            sequence = []
            for seq in range(len(attn_scores[0])):
                head_probs = attn_scores[batch][seq]
                exp_probs = [math.exp(prob) for prob in head_probs]
                sum_exp_probs = sum(exp_probs)
                softmax_probs = [exp_prob / sum_exp_probs for exp_prob in exp_probs]
                sequence.append(softmax_probs)
            softmax_tensor.append(sequence)
            
        output=[]
        for batch in range(len(softmax_tensor)):
            output.append(self.mat_mul(softmax_tensor[batch], V[batch]))
        if list_output is False:
            output=torch.tensor(output)
        return output

         
        
    def mat_mul(self, A, B):
        tensor_A=torch.tensor(A)
        tensor_B=torch.tensor(B)
        assert len(A[0]) == len(B), "The matrices' dimensions are not compatible for multiplication."
        output = [[0] * len(B[0]) for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    output[i][j] += A[i][k] * B[k][j]
        return output

    def transpose(self, array):
        rows = []
        for i in range(len(array[0])):
            cols = []
            for j in range(len(array)):
                cols.append(array[j][i])
            rows.append(cols)
        return rows

    def split_heads(self, x, list_output=True):
        # Reference Code
        # batch_size, seq_length, d_model = x.size()
        # return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        # Finish what was started
        # print("synchronization_12: There is a different message for each synchronization")
        # get the dimensions of the array
        batch_size = len(x)
        seq_length=len(x[0])
        d_model = len(x[0][0])
        # create an empty nested list with the desired dimensions
        output = []
        # iterate through each array in the batch
        for batch in range(batch_size):
            # iterate through each sequence
            sequence = []
            for seq in range(seq_length):
                # iterate through each head
                heads = []
                for head in range(self.num_heads):
                    # take out the corresponding slice of the array
                    sliced_array = (
                        x[batch][seq][head * self.d_k : (head) * self.d_k + self.d_k]
                    ).tolist()
                    # append the slice to the output
                    heads.append(sliced_array)
                sequence.append(heads)
            output.append(self.transpose(sequence))
        # convert the nested list to a tensor
        if list_output is False:
            output=torch.tensor(output)
        return output

    def combine_heads(self, x, list_output=True):
        batch_size=len(x)
        heads=len(x[0]) 
        seq_length=len(x[0][0])
        d_k = len(x[0][0][0])
        result = []
        x = x.tolist()
        for batch in range(batch_size):
            x[batch] = self.transpose(x[batch])
        print(len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]))
        output = []
        for batch in range(batch_size):
            output_sequence = []
            for seq in range(seq_length):
                head_array = []
                for head in range(heads):
                    for k in range(d_k):
                        head_array.append(x[batch][seq][head][k])
                output_sequence.append(head_array)
            output.append(output_sequence)
        print(len(output), len(output[0]), len(output[0][0]))
        if list_output is False:
            output=torch.tensor(output)
        return output

    def forward(self, Q, K, V, list_output=True):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        tensor_Q=torch.tensor(Q)
        tensor_K=torch.tensor(K)
        tensor_V=torch.tensor(V)
        print(tensor_Q.size())
        print(tensor_K.size())
        print(tensor_V.size())
            
        attn_output = self.scaled_attention(Q, K, V)
        tensor_attn=torch.tensor(attn_output)
        print(tensor_attn.size())
        computation=self.combine_heads(torch.tensor(attn_output))
        computation=torch.tensor(computation)
        print(computation)
        output = self.W_o(computation)
        output=output.tolist()
        if list_output is False:
            output=torch.tensor(output)
        print(output)
        return output