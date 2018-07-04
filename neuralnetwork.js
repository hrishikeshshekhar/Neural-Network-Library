function Nn(inputno, hiddenno, outputno)
{
  this.input_nodes = inputno;
  this.hidden_nodes = hiddenno;
  this.output_nodes = outputno;
  this.weights_ih;
  this.weights_ho;
  this.bias_h;
  this.bias_o;

  this.setup = function()
  {
    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weights_ih.setup();
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
    this.weights_ho.setup();
    this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_h.setup();
    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_o.setup();
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

    //Changing the target to a martix
    targets = Matrix.tomatrix(targets);

    //Computing the output error
    error_o = Matrix.subtract(targets, outputs);

    //Computing the error of the hidden layer
    var error_h = new Matrix(this.hidden_nodes, 1);
    error_h.setup();

    //Assigning the errors of the hidden layer
    for(var i = 0; i < this.weights_ho.cols; ++i)
    {
      //Initializing to 0
      error_h.matrix[i][0] = 0;

      for(var j = 0; j < this.weights_ho.rows; ++j)
      {
        error_h.matrix[i][0] += error_o.matrix[j] * this.weights_ho.matrix[j][i] / this.weights_ho.sumofrow(j);
      }
    }
    

  }
}
