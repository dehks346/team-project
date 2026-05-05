import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from bookings import views as booking_views
from bookings.models import Room, Booking, BookingInvitation, Organisation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Face_Recognition.model_state import (
	build_dataset_fingerprint,
	dataset_requires_retraining,
	load_training_state,
	save_training_state,
)


class FaceDatasetStateTests(SimpleTestCase):
	def test_fingerprint_changes_when_new_image_is_added(self):
		with TemporaryDirectory() as temp_dir:
			dataset_dir = Path(temp_dir) / "dataset"
			person_dir = dataset_dir / "alice"
			person_dir.mkdir(parents=True)

			(person_dir / "1.jpg").write_bytes(b"image-one")
			first_signature = build_dataset_fingerprint(dataset_dir)["dataset_signature"]

			(person_dir / "2.jpg").write_bytes(b"image-two")
			second_signature = build_dataset_fingerprint(dataset_dir)["dataset_signature"]

			self.assertNotEqual(first_signature, second_signature)

	def test_non_image_files_are_ignored(self):
		with TemporaryDirectory() as temp_dir:
			dataset_dir = Path(temp_dir) / "dataset"
			person_dir = dataset_dir / "bob"
			person_dir.mkdir(parents=True)

			(person_dir / "1.jpg").write_bytes(b"image-one")
			baseline = build_dataset_fingerprint(dataset_dir)["dataset_signature"]

			(person_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
			after_text_file = build_dataset_fingerprint(dataset_dir)["dataset_signature"]

			self.assertEqual(baseline, after_text_file)

	def test_saved_state_controls_retraining_decision(self):
		with TemporaryDirectory() as temp_dir:
			dataset_dir = Path(temp_dir) / "dataset"
			state_path = Path(temp_dir) / "face_model_state.json"
			person_dir = dataset_dir / "carol"
			person_dir.mkdir(parents=True)
			(person_dir / "1.jpg").write_bytes(b"image-one")

			current_state = build_dataset_fingerprint(dataset_dir)
			save_training_state(state_path, current_state)

			self.assertEqual(load_training_state(state_path)["dataset_signature"], current_state["dataset_signature"])
			self.assertFalse(dataset_requires_retraining(dataset_dir, state_path))

			(person_dir / "2.jpg").write_bytes(b"image-two")
			self.assertTrue(dataset_requires_retraining(dataset_dir, state_path))


class RegistrationFlowTests(TestCase):
	def setUp(self):
		self.leeds_org = Organisation.objects.create(
			name='Leeds Beckett University',
			email_address='admin@leedsbeckett.ac.uk',
			password='hashed-placeholder',
			unique_access_code='LBU-REG-01',
			phone_number='0113-111-1111',
			address='Leeds',
			fee='0.00',
		)
		self.other_org = Organisation.objects.create(
			name='Acme Labs',
			email_address='admin@acmelabs.com',
			password='hashed-placeholder',
			unique_access_code='ACME-REG-01',
			phone_number='0207-111-1111',
			address='London',
			fee='0.00',
		)

	def test_register_creates_user_via_website(self):
		response = self.client.post(
			reverse('register'),
			{
				'name': 'Alex Chen',
				'email': 'alex@example.com',
				'password1': 'StrongPass123!',
				'password2': 'StrongPass123!',
				'role': 'user',
			},
		)

		self.assertEqual(response.status_code, 302)
		self.assertTrue(User.objects.filter(email='alex@example.com').exists())

		created_user = User.objects.get(email='alex@example.com')
		self.assertEqual(created_user.first_name, 'Alex')
		self.assertEqual(created_user.last_name, 'Chen')
		self.assertFalse(created_user.is_superuser)

	def test_register_auto_assigns_organisation_from_email_domain(self):
		response = self.client.post(
			reverse('register'),
			{
				'name': 'Hamdi Student',
				'email': 'hamdi@leedsbeckett.ac.uk',
				'password1': 'StrongPass123!',
				'password2': 'StrongPass123!',
				'role': 'user',
				'organisation': '',
			},
		)

		self.assertEqual(response.status_code, 302)
		created_user = User.objects.get(email='hamdi@leedsbeckett.ac.uk')
		self.assertEqual(created_user.userprofile.organisation, self.leeds_org)

	def test_register_manual_organisation_overrides_domain_match(self):
		response = self.client.post(
			reverse('register'),
			{
				'name': 'Taylor Example',
				'email': 'taylor@leedsbeckett.ac.uk',
				'password1': 'StrongPass123!',
				'password2': 'StrongPass123!',
				'role': 'user',
				'organisation': str(self.other_org.pk),
			},
		)

		self.assertEqual(response.status_code, 302)
		created_user = User.objects.get(email='taylor@leedsbeckett.ac.uk')
		self.assertEqual(created_user.userprofile.organisation, self.other_org)

	def test_non_admin_cannot_create_admin_account(self):
		response = self.client.post(
			reverse('register'),
			{
				'name': 'Eve Admin',
				'email': 'eve@example.com',
				'password1': 'StrongPass123!',
				'password2': 'StrongPass123!',
				'role': 'admin',
			},
		)

		self.assertEqual(response.status_code, 200)
		self.assertFalse(User.objects.filter(email='eve@example.com').exists())


class LoginRedirectTests(TestCase):
	def test_login_redirects_to_dashboard(self):
		User.objects.create_user(
			username='loginuser',
			email='login@example.com',
			password='StrongPass123!',
		)

		response = self.client.post(
			reverse('login'),
			{
				'username': 'loginuser',
				'password': 'StrongPass123!',
			},
		)

		self.assertEqual(response.status_code, 302)
		self.assertEqual(response.url, reverse('home'))

	def test_login_with_email_redirects_to_dashboard(self):
		User.objects.create_user(
			username='emailuser',
			email='emailuser@example.com',
			password='StrongPass123!',
		)

		response = self.client.post(
			reverse('login'),
			{
				'username': 'emailuser@example.com',
				'password': 'StrongPass123!',
			},
		)

		self.assertEqual(response.status_code, 302)
		self.assertEqual(response.url, reverse('home'))

	def test_face_login_redirects_to_dashboard(self):
		User.objects.create_user(
			username='faceuser',
			email='face@example.com',
			password='StrongPass123!',
		)

		booking_views.confirmed_name = 'faceuser'
		booking_views.frame_count = booking_views.REQUIRED_CONSISTENT_FRAMES
		booking_views.last_prediction_distance = 30

		response = self.client.post(reverse('complete_face_login'))

		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.json()['redirect_url'], reverse('home'))


class FaceRequiredBookingFlowTests(TestCase):
	def setUp(self):
		self.user = User.objects.create_user(
			username='bookinguser',
			email='booking@example.com',
			password='StrongPass123!',
		)
		self.client.force_login(self.user)

	def test_face_required_room_redirects_to_face_verification(self):
		response = self.client.get(reverse('booking_face_gate', args=['conference-a']))

		self.assertEqual(response.status_code, 302)
		self.assertIn(reverse('face_verification'), response.url)

	def test_non_face_required_room_goes_directly_to_booking_form(self):
		response = self.client.get(reverse('booking_face_gate', args=['meeting-pod-b']))

		self.assertEqual(response.status_code, 302)
		self.assertEqual(response.url, f"{reverse('booking_creation')}?room=meeting-pod-b")

	def test_booking_page_is_protected_for_face_required_room(self):
		response = self.client.get(f"{reverse('booking_creation')}?room=conference-a")

		self.assertEqual(response.status_code, 302)
		self.assertIn(reverse('face_verification'), response.url)

	def test_face_login_redirects_back_to_booking_form_when_pending(self):
		self.client.get(reverse('booking_face_gate', args=['conference-a']))

		booking_views.confirmed_name = 'bookinguser'
		booking_views.frame_count = booking_views.REQUIRED_CONSISTENT_FRAMES
		booking_views.last_prediction_distance = 25

		response = self.client.post(reverse('complete_face_login'))

		self.assertEqual(response.status_code, 200)
		self.assertEqual(
			response.json()['redirect_url'],
			f"{reverse('booking_creation')}?room=conference-a",
		)


class RoomAndBookingPersistenceTests(TestCase):
	def setUp(self):
		self.user = User.objects.create_user(
			username='hostuser',
			email='host@example.com',
			password='StrongPass123!',
		)
		self.invited_user = User.objects.create_user(
			username='guestuser',
			email='guest@example.com',
			password='StrongPass123!',
		)
		self.client.force_login(self.user)

	def test_room_creation_saves_to_database(self):
		response = self.client.post(
			reverse('room_creation'),
			{
				'name': 'Innovation Lab',
				'location': 'Floor 3 West Wing',
				'capacity': '14',
				'room_type': 'CONF',
				'is_face_required': 'on',
				'approval_required': 'on',
				'equipment': 'TV, Whiteboard',
				'opening_time': '08:00',
				'closing_time': '18:00',
			},
		)

		self.assertEqual(response.status_code, 302)
		self.assertTrue(Room.objects.filter(name='Innovation Lab', location='Floor 3 West Wing').exists())

	def test_booking_creation_saves_booking_and_invites(self):
		held_room = Room.objects.create(
			name='Team Room',
			location='Level 2',
			capacity=8,
			room_type='MEET',
			is_face_required=False,
		)

		response = self.client.post(
			f"{reverse('booking_creation')}?room={held_room.room_id}",
			{
				'room': str(held_room.room_id),
				'booking_date': '2030-01-01',
				'start_time': '10:00',
				'duration': '60',
				'notes': 'Planning session',
				'recurring': 'on',
				'recurrence_pattern': 'Every week',
				'invitees': 'guest@example.com',
			},
		)

		self.assertEqual(response.status_code, 302)
		self.assertTrue(Booking.objects.filter(room=held_room, notes='Planning session').exists())
		booking = Booking.objects.get(room=held_room, notes='Planning session')
		self.assertTrue(BookingInvitation.objects.filter(booking=booking, invited_email='guest@example.com').exists())

	def test_invitation_response_is_saved(self):
		held_room = Room.objects.create(
			name='Team Room',
			location='Level 2',
			capacity=8,
			room_type='MEET',
			is_face_required=False,
		)
		booking = Booking.objects.create(
			user=self.user.userprofile,
			room=held_room,
			booking_datetime=booking_views.timezone.make_aware(booking_views.datetime(2030, 1, 1, 10, 0)),
		)
		invite = BookingInvitation.objects.create(booking=booking, invited_user=self.invited_user, invited_email='guest@example.com')

		self.client.logout()
		self.client.force_login(self.invited_user)
		response = self.client.post(reverse('booking_invitation_respond', args=[invite.invitation_id]), {'action': 'accept'})

		self.assertEqual(response.status_code, 302)
		invite.refresh_from_db()
		self.assertEqual(invite.status, 'ACCEPTED')


class OrganisationAccessTests(TestCase):
	def setUp(self):
		self.organisation = Organisation.objects.create(
			name='Leeds Beckett University',
			email_address='admin@leedsbeckett.ac.uk',
			password='hashed-placeholder',
			unique_access_code='LBU-001',
			phone_number='0113-000-0000',
			address='Leeds, UK',
			fee='0.00',
		)

	def test_organisation_user_can_create_room_for_their_org(self):
		org_user = User.objects.create_user(
			username='orguser',
			email='orguser@leedsbeckett.ac.uk',
			password='StrongPass123!',
		)
		org_user.userprofile.organisation = self.organisation
		org_user.userprofile.save(update_fields=['organisation'])

		self.client.force_login(org_user)
		response = self.client.post(
			reverse('room_creation'),
			{
				'name': 'Org Study Room',
				'location': 'Campus Building A',
				'capacity': '20',
				'room_type': 'MEET',
			},
		)

		self.assertEqual(response.status_code, 302)
		created_room = Room.objects.get(name='Org Study Room')
		self.assertEqual(created_room.organisation, self.organisation)

	def test_regular_user_cannot_create_room(self):
		regular_user = User.objects.create_user(
			username='regularuser',
			email='regular@example.com',
			password='StrongPass123!',
		)

		self.client.force_login(regular_user)
		response = self.client.post(
			reverse('room_creation'),
			{
				'name': 'Should Not Exist',
				'location': 'Campus Building B',
				'capacity': '10',
				'room_type': 'MEET',
			},
		)

		self.assertNotEqual(response.status_code, 200)
		self.assertFalse(Room.objects.filter(name='Should Not Exist').exists())
