using System;
using static NNLib.NN;

namespace NNLib
{
    internal class Program
    {
        private static void Main(string[] args)
        {

            Layer layer1 = new Layer(new Neurons.SqrtNeuron(), 2);
            Layer layer2 = new Layer(new Neurons.SqrtNeuron(), 5);
            Layer layer3 = new Layer(new Neurons.ReLUNeuron(), 3);
            Layer layer4 = new Layer(new Neurons.SqrtNeuron(), 1);
            NeuralNetwork nn = new NeuralNetwork(layer1, layer2, layer3, layer4);

            Console.WriteLine(nn.getWeights().Length);
            Data.Dataset dataset0 = new Data.Dataset(new double[2] { 0, 0 }, new double[1] { 1 });
            Data.Dataset dataset1 = new Data.Dataset(new double[2] { 1, 0 }, new double[1] { 2 });
            Data.Dataset dataset2 = new Data.Dataset(new double[2] { 2, 1 }, new double[1] { 4 });
            Data.Dataset dataset3 = new Data.Dataset(new double[2] { 3, 1 }, new double[1] { 5 });
            Data.Dataset dataset4 = new Data.Dataset(new double[2] { 4, 0 }, new double[1] { 5 });
            Data.DataCollection dataCollection = new Data.DataCollection(dataset0, dataset1, dataset2, dataset3, dataset4);

            Basis.Trainer trainer = new NN.Trainer.GeneticTrainer();

            nn = trainer.train(nn, dataCollection, new Loss.SquaredLoss(), 30);

            Console.WriteLine(nn.Calculate(new double[2] { 1, 0 })[0].value);
            string weights = "";
            foreach (double d in nn.getWeights())
            {
                weights += d + " ";
            }
            Console.WriteLine(weights);
            Console.ReadLine();

        }
    }
}
