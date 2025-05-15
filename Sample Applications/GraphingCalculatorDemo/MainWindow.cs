using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace GraphingCalculatorDemo
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void PlotExample_Click(object sender, RoutedEventArgs e)
        {
            PlotCanvas.Children.Clear();

            // Draw axes
            var axis = new Line
            {
                X1 = 20, Y1 = 100,
                X2 = 380, Y2 = 100,
                Stroke = Brushes.Black,
                StrokeThickness = 2
            };
            PlotCanvas.Children.Add(axis);

            // Draw a sine wave as an example
            Polyline polyline = new Polyline
            {
                Stroke = Brushes.Blue,
                StrokeThickness = 2
            };

            for (int x = 0; x <= 360; x += 5)
            {
                double radians = x * System.Math.PI / 180;
                double y = 100 - 80 * System.Math.Sin(radians);
                polyline.Points.Add(new Point(20 + x * 1.0, y));
            }

            PlotCanvas.Children.Add(polyline);
        }
    }
}