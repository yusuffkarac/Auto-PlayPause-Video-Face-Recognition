import unittest
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
import sys
import os

# Mock face_recognition and pynput modules before importing recognition
sys.modules['face_recognition'] = MagicMock()
sys.modules['pynput'] = MagicMock()
sys.modules['pynput.keyboard'] = MagicMock()

# Import the modules we want to test
from recognition import face_confidence, FaceRecognition


class TestFaceConfidence(unittest.TestCase):
    """Test cases for the face_confidence helper function"""

    def test_high_confidence_match(self):
        """Test face_confidence with a very close match (low distance)"""
        result = face_confidence(0.3)
        # Should return a high percentage
        self.assertIn('%', result)
        confidence_value = float(result.replace('%', ''))
        self.assertGreater(confidence_value, 80)

    def test_low_confidence_match(self):
        """Test face_confidence with a poor match (high distance)"""
        result = face_confidence(0.8)
        # Should return a low percentage
        self.assertIn('%', result)
        confidence_value = float(result.replace('%', ''))
        self.assertLess(confidence_value, 50)

    def test_perfect_match(self):
        """Test face_confidence with a perfect match (distance = 0)"""
        result = face_confidence(0.0)
        self.assertIn('%', result)
        confidence_value = float(result.replace('%', ''))
        self.assertGreater(confidence_value, 95)

    def test_threshold_boundary(self):
        """Test face_confidence at the threshold boundary"""
        result = face_confidence(0.6)
        self.assertIn('%', result)
        # Should return a valid percentage string
        confidence_value = float(result.replace('%', ''))
        self.assertGreaterEqual(confidence_value, 0)
        self.assertLessEqual(confidence_value, 100)

    def test_custom_threshold(self):
        """Test face_confidence with a custom threshold"""
        result = face_confidence(0.5, face_match_threshold=0.7)
        self.assertIn('%', result)
        confidence_value = float(result.replace('%', ''))
        self.assertGreaterEqual(confidence_value, 0)
        self.assertLessEqual(confidence_value, 100)


class TestFaceRecognitionInit(unittest.TestCase):
    """Test cases for FaceRecognition class initialization"""

    def setUp(self):
        """Reset class variables before each test"""
        FaceRecognition.known_face_encodings = []
        FaceRecognition.known_face_names = []

    @patch('recognition.os.listdir')
    @patch('recognition.face_recognition.load_image_file')
    @patch('recognition.face_recognition.face_encodings')
    def test_initialization(self, mock_encodings, mock_load_image, mock_listdir):
        """Test that FaceRecognition initializes correctly"""
        # Mock the faces directory
        mock_listdir.return_value = ['test_face.jpg']
        mock_load_image.return_value = np.array([[[1, 2, 3]]])
        mock_encodings.return_value = [np.array([0.1, 0.2, 0.3])]

        fr = FaceRecognition()

        # Check that initial state is correct
        self.assertEqual(fr.is_video_active, False)
        self.assertEqual(fr.goo, True)
        self.assertEqual(fr.process_current_frame, True)
        self.assertIsInstance(fr.face_locations, list)
        self.assertIsInstance(fr.face_encodings, list)
        self.assertIsInstance(fr.face_names, list)

    @patch('recognition.os.listdir')
    @patch('recognition.face_recognition.load_image_file')
    @patch('recognition.face_recognition.face_encodings')
    def test_initialization_calls_encode_faces(self, mock_encodings, mock_load_image, mock_listdir):
        """Test that initialization calls encode_faces"""
        mock_listdir.return_value = ['face1.jpg']
        mock_load_image.return_value = np.array([[[1, 2, 3]]])
        mock_encodings.return_value = [np.array([0.1, 0.2, 0.3])]

        fr = FaceRecognition()

        # Verify encode_faces was called
        mock_listdir.assert_called_once_with('faces')
        self.assertEqual(len(fr.known_face_names), 1)
        self.assertEqual(len(fr.known_face_encodings), 1)


class TestEncodeFaces(unittest.TestCase):
    """Test cases for the encode_faces method"""

    def setUp(self):
        """Reset class variables before each test"""
        FaceRecognition.known_face_encodings = []
        FaceRecognition.known_face_names = []

    @patch('recognition.os.listdir')
    @patch('recognition.face_recognition.load_image_file')
    @patch('recognition.face_recognition.face_encodings')
    def test_encode_single_face(self, mock_encodings, mock_load_image, mock_listdir):
        """Test encoding a single face"""
        mock_listdir.return_value = ['person1.jpg']
        mock_load_image.return_value = np.array([[[1, 2, 3]]])
        test_encoding = np.array([0.1, 0.2, 0.3])
        mock_encodings.return_value = [test_encoding]

        fr = FaceRecognition()

        self.assertEqual(len(fr.known_face_encodings), 1)
        self.assertEqual(len(fr.known_face_names), 1)
        self.assertEqual(fr.known_face_names[0], 'person1.jpg')
        np.testing.assert_array_equal(fr.known_face_encodings[0], test_encoding)

    @patch('recognition.os.listdir')
    @patch('recognition.face_recognition.load_image_file')
    @patch('recognition.face_recognition.face_encodings')
    def test_encode_multiple_faces(self, mock_encodings, mock_load_image, mock_listdir):
        """Test encoding multiple faces"""
        mock_listdir.return_value = ['person1.jpg', 'person2.jpg', 'person3.png']
        mock_load_image.return_value = np.array([[[1, 2, 3]]])
        mock_encodings.return_value = [np.array([0.1, 0.2, 0.3])]

        fr = FaceRecognition()

        self.assertEqual(len(fr.known_face_encodings), 3)
        self.assertEqual(len(fr.known_face_names), 3)
        self.assertIn('person1.jpg', fr.known_face_names)
        self.assertIn('person2.jpg', fr.known_face_names)
        self.assertIn('person3.png', fr.known_face_names)

    @patch('recognition.os.listdir')
    def test_encode_empty_directory(self, mock_listdir):
        """Test encoding when faces directory is empty"""
        mock_listdir.return_value = []

        fr = FaceRecognition()

        self.assertEqual(len(fr.known_face_encodings), 0)
        self.assertEqual(len(fr.known_face_names), 0)


class TestRunRecognition(unittest.TestCase):
    """Test cases for the run_recognition method"""

    def setUp(self):
        """Reset class variables before each test"""
        FaceRecognition.known_face_encodings = []
        FaceRecognition.known_face_names = []

    @patch('recognition.cv2.VideoCapture')
    @patch('recognition.os.listdir')
    @patch('recognition.face_recognition.load_image_file')
    @patch('recognition.face_recognition.face_encodings')
    def test_video_source_not_found(self, mock_encodings, mock_load_image, mock_listdir, mock_video_capture):
        """Test that sys.exit is called when video source is not available"""
        mock_listdir.return_value = []

        # Mock video capture to fail
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = False
        mock_video_capture.return_value = mock_capture_instance

        fr = FaceRecognition()

        with self.assertRaises(SystemExit) as cm:
            fr.run_recognition()

        self.assertEqual(str(cm.exception), 'Video source not found...')

    @patch('recognition.cv2.VideoCapture')
    @patch('recognition.cv2.resize')
    @patch('recognition.cv2.imshow')
    @patch('recognition.cv2.waitKey')
    @patch('recognition.cv2.destroyAllWindows')
    @patch('recognition.face_recognition.face_locations')
    @patch('recognition.face_recognition.face_encodings')
    @patch('recognition.os.listdir')
    @patch('recognition.face_recognition.load_image_file')
    def test_quit_on_q_key(self, mock_load_image, mock_listdir, mock_face_encodings_func,
                           mock_face_locations, mock_destroy, mock_waitkey,
                           mock_imshow, mock_resize, mock_video_capture):
        """Test that pressing 'q' quits the application"""
        mock_listdir.return_value = []

        # Mock video capture
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture_instance.read.return_value = (True, mock_frame)
        mock_video_capture.return_value = mock_capture_instance

        # Mock face recognition functions
        mock_face_locations.return_value = []
        mock_face_encodings_func.return_value = []

        # Mock resize
        mock_resize.return_value = np.zeros((120, 160, 3), dtype=np.uint8)

        # Simulate pressing 'q' key after one iteration
        mock_waitkey.return_value = ord('q')

        fr = FaceRecognition()
        fr.run_recognition()

        # Verify video capture was released and windows destroyed
        mock_capture_instance.release.assert_called_once()
        mock_destroy.assert_called_once()

    @patch('recognition.cv2.VideoCapture')
    @patch('recognition.cv2.resize')
    @patch('recognition.cv2.imshow')
    @patch('recognition.cv2.waitKey')
    @patch('recognition.cv2.destroyAllWindows')
    @patch('recognition.cv2.rectangle')
    @patch('recognition.cv2.putText')
    @patch('recognition.face_recognition.face_locations')
    @patch('recognition.face_recognition.face_encodings')
    @patch('recognition.face_recognition.compare_faces')
    @patch('recognition.face_recognition.face_distance')
    @patch('recognition.Controller')
    @patch('recognition.os.listdir')
    @patch('recognition.face_recognition.load_image_file')
    @patch('recognition.time.sleep')
    def test_face_detection_triggers_play(self, mock_sleep, mock_load_image, mock_listdir,
                                         mock_controller, mock_face_distance, mock_compare_faces,
                                         mock_face_encodings_func, mock_face_locations,
                                         mock_puttext, mock_rectangle, mock_destroy,
                                         mock_waitkey, mock_imshow, mock_resize, mock_video_capture):
        """Test that detecting a face triggers video play"""
        # Setup known faces
        mock_listdir.return_value = ['test.jpg']
        mock_load_image.return_value = np.array([[[1, 2, 3]]])
        mock_face_encodings_func.side_effect = [
            [np.array([0.1, 0.2, 0.3])],  # For encode_faces
            [np.array([0.1, 0.2, 0.3])]   # For run_recognition
        ]

        # Mock video capture
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture_instance.read.return_value = (True, mock_frame)
        mock_video_capture.return_value = mock_capture_instance

        # Mock face detection - face found
        mock_face_locations.return_value = [(10, 20, 30, 40)]
        mock_compare_faces.return_value = [True]
        mock_face_distance.return_value = np.array([0.3])

        # Mock resize
        mock_resize.return_value = np.zeros((120, 160, 3), dtype=np.uint8)

        # Mock keyboard controller
        mock_kb_instance = Mock()
        mock_controller.return_value = mock_kb_instance

        # Quit after first iteration
        mock_waitkey.return_value = ord('q')

        fr = FaceRecognition()
        fr.run_recognition()

        # Verify keyboard was pressed (to play video)
        self.assertTrue(mock_kb_instance.press.called)
        self.assertTrue(mock_kb_instance.release.called)

    @patch('recognition.cv2.VideoCapture')
    @patch('recognition.cv2.resize')
    @patch('recognition.cv2.imshow')
    @patch('recognition.cv2.waitKey')
    @patch('recognition.cv2.destroyAllWindows')
    @patch('recognition.face_recognition.face_locations')
    @patch('recognition.face_recognition.face_encodings')
    @patch('recognition.Controller')
    @patch('recognition.os.listdir')
    @patch('recognition.face_recognition.load_image_file')
    @patch('recognition.time.sleep')
    def test_no_face_detection_triggers_pause(self, mock_sleep, mock_load_image, mock_listdir,
                                              mock_controller, mock_face_encodings_func,
                                              mock_face_locations, mock_destroy,
                                              mock_waitkey, mock_imshow, mock_resize,
                                              mock_video_capture):
        """Test that not detecting a face triggers video pause"""
        mock_listdir.return_value = []

        # Mock video capture
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture_instance.read.return_value = (True, mock_frame)
        mock_video_capture.return_value = mock_capture_instance

        # Mock face detection - no faces found
        mock_face_locations.return_value = []
        mock_face_encodings_func.return_value = []

        # Mock resize
        mock_resize.return_value = np.zeros((120, 160, 3), dtype=np.uint8)

        # Mock keyboard controller
        mock_kb_instance = Mock()
        mock_controller.return_value = mock_kb_instance

        # Quit after first iteration
        mock_waitkey.return_value = ord('q')

        fr = FaceRecognition()
        fr.is_video_active = True  # Simulate video already playing
        fr.run_recognition()

        # Verify keyboard was pressed (to pause video)
        self.assertTrue(mock_kb_instance.press.called)
        self.assertTrue(mock_kb_instance.release.called)


if __name__ == '__main__':
    unittest.main()
