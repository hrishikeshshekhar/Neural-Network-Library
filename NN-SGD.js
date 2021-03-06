function Nn(inputno, hiddenno, outputno)
{
  this.input_nodes = inputno;
  this.hidden_nodes = hiddenno;
  this.output_nodes = outputno;
  this.learning_rate = 0.1;
  this.weights_ih;
  this.weights_ho;
  this.bias_h;
  this.bias_o;

  this.setup = function(learning_rate)
  {
    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weights_ih.setup();
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
    this.weights_ho.setup();
    this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_h.setup();
    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_o.setup();
    this.learning_rate = learning_rate;
  }

  this.feedforward = function(inputs)
  {
    //Creating the output matrix
    var output = new Matrix(inputs.length, 1);

    //Converting the input into a matrix
    inputs = Matrix.tomatrix(inputs);

    //Generating the hidden layer output
    output = Matrix.multiply(this.weights_ih, inputs);
    output = Matrix.add(output, this.bias_h);
    //Passing output through activation function
    output.activate();

    //Generating the final output
    output = Matrix.multiply(this.weights_ho, output);
    output = Matrix.add(output, this.bias_o);
    //Passing output through activation function
    output.activate();

    //Changing the output into an array
    output = output.toarray();

    //Returning the output
    return output;
  }

  this.train = function(inputs, targets)
  {
    //Creating the output errors
    var error_o = new Matrix(targets.length, 1);

    var outputs = this.feedforward(inputs);

    //Converting the output into a matrix
    outputs = Matrix.tomatrix(outputs);

    //Converting the inputs to a matrix
    inputs = Matrix.tomatrix(inputs);

    //Changing the target to a martix
    targets = Matrix.tomatrix(targets);

    //Computing the output error
    error_o = Matrix.subtract(targets, outputs);

    //Normailzing the weights in the hidden layer
    var n_weights_ho = new Matrix(this.weights_ho.rows, this.weights_ho.cols);
    n_weights_ho.setup();

    //Assigning the normalized weights matrix
    for(var i = 0; i < n_weights_ho.rows; ++i)
    {
      var rowsum = this.weights_ho.sumofrow(i);
      for(var j = 0; j < n_weights_ho.cols; ++j)
      {
        n_weights_ho.matrix[i][j] = this.weights_ho.matrix[i][j] / rowsum;
      }
    }

    //Transposing this normalised weights matrix
    var n_weights_ho_t = Matrix.transpose(n_weights_ho);

    //Finding the error in the hidden layer
    var error_h = Matrix.multiply(n_weights_ho_t, error_o);

    //Computing the change in weights and bias in the hidden layer and assigning them
    var hidden_inputs = Matrix.multiply(this.weights_ih, inputs);
    hidden_inputs = Matrix.add(hidden_inputs, this.bias_h);
    hidden_inputs.activate();

    //Finding change in weights in output, hidden layer
    outputs.activateder();
    var gradient_ho = Matrix.hadmardproduct(outputs, error_o);
    var delta_w_ho = Matrix.multiply(gradient_ho, Matrix.transpose(hidden_inputs));

    //Multiplying the change by learning rate
    delta_w_ho.multiplyscaler(this.learning_rate);
    this.weights_ho = Matrix.add(this.weights_ho, delta_w_ho);

    //Changing the biases in the output layer
    gradient_ho.multiplyscaler(this.learning_rate);
    this.bias_o = Matrix.add(this.bias_o, gradient_ho);

    //Computing the change in weights of the input layer
    hidden_inputs.activateder();
    var gradient_ih = Matrix.hadmardproduct(hidden_inputs, error_h);
    var delta_w_ih = Matrix.multiply(gradient_ih, Matrix.transpose(inputs));

    //Multipling by learning rate
    delta_w_ih.multiplyscaler(this.learning_rate);
    this.weights_ih = Matrix.add(this.weights_ih, delta_w_ih);

    //Changing the biases in the hidden layer
    gradient_ih.multiplyscaler(this.learning_rate);
    this.bias_h = Matrix.add(this.bias_h, gradient_ih);
  }

  //Function to predict given neural network
  this.predict = function(inputs)
  {
    return (this.feedforward(inputs));
  }
}