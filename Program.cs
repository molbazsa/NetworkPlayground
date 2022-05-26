using MathNet.Numerics.LinearAlgebra;

Vector<double> VectorAppend(Vector<double> vector, double value) {
    return Vector<double>.Build.Dense(
        vector.Append(value).ToArray()
    );
}

Vector<double> GenerateTrainingData(
    Vector<double> input,
    List<Matrix<double>> weightMatrices
) {
    foreach (var weightMatrix in weightMatrices) {
        input = weightMatrix * VectorAppend(input, 1);
    }

    return input;
}

var x = Vector<double>.Build.Dense(
    new double[] { 1, 2 }
);

var v = Matrix<double>.Build.DenseOfArray(
    new double[,] {
        { 1, 2, 1 },
        { 2, 2, 1 },
    }
);

var w = Matrix<double>.Build.DenseOfArray(
    new double[,] {
        { 2, 1, 0 },
        { 1, 1, 1 },
    }
);

var z = GenerateTrainingData(x, new List<Matrix<double>> { v, w });

Console.WriteLine(x);
Console.WriteLine(z);
