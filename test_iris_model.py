
import unittest

from iris_model import load_dataset, train_model


class TestIrisModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_dataset()
        (
            cls.model,
            cls.X_test,
            cls.y_test,
            cls.accuracy
        ) = train_model()

    def test_dataset_has_150_records(self):
        self.assertEqual(len(self.X), 150)

    def test_dataset_has_four_features(self):
        self.assertEqual(self.X.shape[1], 4)

    def test_dataset_has_three_species(self):
        self.assertEqual(self.y.nunique(), 3)

    def test_dataset_has_no_missing_values(self):
        self.assertEqual(
            int(self.X.isnull().sum().sum()), 0
        )

    def test_model_predicts_valid_species(self):
        prediction = self.model.predict(
            self.X_test.iloc[[0]]
        )[0]

        self.assertIn(prediction, self.y.unique())

    def test_model_accuracy(self):
        self.assertGreaterEqual(self.accuracy, 0.90)


if __name__ == "__main__":
    unittest.main()
