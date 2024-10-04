# DocTech Application

DocTech is an advanced diagnostic web application that allows users to diagnose issues related to PCs, laptops, tablets, and phones through a chatbot interface. The application is designed to provide an intuitive and efficient user experience with modern web technologies. It features a robust authentication system, news updates, and a "Talk to Us" feature for users to communicate directly with support.

## Table of Contents
- [Live Demo](#live-demo)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Frontend](#frontend)
  - [Backend](#backend)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Live Demo
The application is live and can be accessed through the following links:

- **Frontend (Next.js on Vercel):** [DocTech Frontend](https://pcairepair.vercel.app/)
- **Backend (Django Rest Framework on Render):** [DocTech API](https://aipcrepair.onrender.com/apis/)

## Features
- **AI-based Diagnostic Chatbot**: Users can interact with the chatbot to diagnose common hardware and software issues.
- **Authentication with JWT**: Secure login and signup flow using JSON Web Tokens (JWT) for protected routes.
- **Profile Management**: Each user can manage their profile, including uploading a profile picture stored on Backblaze.
- **Responsive Design**: The frontend is optimized for a seamless experience across various devices, from mobile to desktop.
- **News Section**: Latest news related to tech is dynamically fetched and displayed, with lazy loading and responsiveness.
- **Email Verification and Welcome Emails**: Secure email verification flow along with a stylish welcome email template.
- **"Talk to Us" Feature**: Users can directly send messages to the support team, pre-filling their name and email for convenience.

## Technologies Used

### Frontend
- **Next.js**: Framework for server-side rendering and static site generation.
- **Axios**: For handling API requests to the backend.
- **Tailwind CSS**: For utility-first responsive and modern styling.
- **JWT Authentication**: For managing user sessions and secure access to protected routes.

### Backend
- **Django**: A high-level Python web framework.
- **Django Rest Framework (DRF)**: For building the API backend.
- **JWT Authentication**: Used for secure token-based authentication.
- **MySQL**: The database used for storing user profiles, chats, and diagnoses.

## Architecture

### Frontend (Next.js)
- Hosted on **Vercel**.
- Handles authentication, state management, and API calls using Axios.
- Responsive user interface designed with Tailwind CSS.

### Backend (Django Rest Framework)
- Hosted on **Render**.
- Provides RESTful APIs for user authentication, chat functionality, and diagnostics.
- Uses JWT for secure user authentication.
- **MySQL** database for storing data.

### Cloud Storage (Backblaze)
- Used for storing user-uploaded profile pictures.

## Getting Started

### Frontend

1. Clone the frontend repository:

   ```bash
   git clone https://github.com/EpheyHertz/pcairepairfrontend.git
   cd pcairepair-frontend
2. Install dependencies:

   ```
   npm install

    
3. Create a .env.local file in the root directory and add the necessary environment variables:

    ```
    NEXT_PUBLIC_API_BASE_URL=https://aipcrepair.onrender.com/apis/
4. Run the development server:

   ```
   npm run dev
   Visit the application at http://localhost:3000.

5. Backend
5.1 Clone the backend repository:


    ```
        git clone https://github.com/EpheyHertz/pcairepairbackend.git
        cd aipcrepair-backend

5.2 Install Python dependencies:

    ```
    pip install -r requirements.txt
5.3 Set up your MySQL database and configure the settings.py file:


        ```
        DATABASES = {
            'default': {
                'ENGINE': 'django.db.backends.mysql',
                'NAME': 'your_db_name',
                'USER': 'your_db_user',
                'PASSWORD': 'your_db_password',
                'HOST': 'your_db_host',
                'PORT': 'your_db_port',
            }
        }
5.4 Set up environment variables for JWT authentication in .env:


    ```
    SECRET_KEY=your_secret_key
5.5 Run the migrations and start the server:

    ```
    
    python manage.py migrate
    python manage.py runserver
Visit the API at http://localhost:8000.

API Documentation
The backend provides several endpoints for user authentication, profile management, chats, and diagnostics. For detailed API usage and documentation, visit the DocTech API Documentation.

Deployment
Frontend Deployment (Vercel)
The frontend is automatically deployed via Vercel. The vercel.json file contains the configuration for deployment. Visit the live application at DocTech Frontend on Vercel.

Backend Deployment (Render)
The backend is hosted on Render, with automatic deployments on push to the main branch. Visit the API live at DocTech API on Render.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch for your feature or bugfix.
Commit your changes.
Open a pull request, and we will review your submission.
Please follow the contribution guidelines for more details.


Acknowledgments
We would like to thank the open-source community for their continuous support and contributions to the tools and frameworks that made DocTech possible.




