# Complete Guide to Setting Up the Production System

This guide provides comprehensive steps to set up and run the Dehazing and Deobstruction System in a production environment, including both the backend server and mobile application.

## Backend Server Production Setup

### Option 1: Running on a Windows Server

1. Clone or download the repository to your production server
2. Install Python 3.8 or higher
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Generate model weights:
   ```
   python generate_weights.py
   ```
5. Run in production mode:
   ```
   python production.py
   ```
   
   The server will run on port 5000 using the Waitress WSGI server, which is suitable for production use on Windows.

### Option 2: Docker Deployment

1. Make sure Docker is installed on your server
2. Build and run the Docker container:
   ```
   docker-compose up -d
   ```
   
   This will create a containerized version of the application that is isolated from the host system.

### Option 3: Cloud Deployment (AWS, Azure, GCP)

1. Set up a cloud VM or app service
2. Clone the repository
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up appropriate environment variables:
   ```
   export FLASK_ENV=production
   export DATABASE_URL=your_database_connection_string
   ```
5. Run with gunicorn (Linux/Mac) or waitress (Windows):
   ```
   # Linux/Mac
   gunicorn -w 4 -b 0.0.0.0:5000 main:app
   
   # Windows
   python -m waitress --port=5000 main:app
   ```

## Mobile App Production Setup

### Prerequisites

- Node.js 14 or higher
- React Native development environment
- Android Studio (for Android build)
- Xcode (for iOS build, macOS only)

### Building the React Native App

1. Navigate to the mobile_app directory:
   ```
   cd mobile_app
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Update the API URL in `src/config/api.js` to point to your production server:
   ```javascript
   const API_URL = 'https://your-production-server.com';
   export { API_URL };
   ```

4. Build for Android:
   ```
   cd android
   ./gradlew assembleRelease
   ```
   
   The APK will be located at `android/app/build/outputs/apk/release/app-release.apk`

5. Build for iOS (requires macOS):
   ```
   cd ios
   pod install
   ```
   
   Then open the .xcworkspace file in Xcode and archive the app for distribution.

## Production Maintenance

### Monitoring

- Set up monitoring using tools like Prometheus, Grafana, or cloud provider monitoring services
- Monitor server CPU, memory, and disk usage
- Track API request volume and response times

### Backups

- Regularly back up the database (located at `instance/dehazing.db` by default)
- Back up model weights in `static/models/weights/`
- Store backups in a secure, off-site location

### Updates

When updating the application:

1. Test changes in a staging environment first
2. Use a deployment strategy that minimizes downtime
3. Consider implementing a CI/CD pipeline for automated testing and deployment

## Security Considerations

1. Use HTTPS for all production deployments
2. Implement rate limiting to prevent abuse
3. Consider adding authentication for API endpoints
4. Regularly update dependencies to patch security vulnerabilities

## Scaling

If your application needs to handle higher load:

1. Use a load balancer to distribute traffic across multiple instances
2. Consider separating the database to its own server
3. Implement caching for frequently accessed results
4. Use a CDN for serving static files and processed images

## Troubleshooting

### Server Issues

- Check logs for error messages
- Verify that model weights are properly loaded
- Ensure sufficient disk space for storing uploaded and processed images/videos

### Mobile App Issues

- Test API connectivity from the mobile device
- Verify that the correct API URL is configured
- Check permissions for camera and photo library access

## Contact and Support

For assistance with deployment issues, contact:
- Email: support@dehazing-system.com
- GitHub: https://github.com/yourname/dehazing-system
