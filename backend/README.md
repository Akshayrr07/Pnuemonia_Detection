# Backend

This directory is reserved for the production Python API.

Planned responsibilities:

- expose health and prediction endpoints
- validate uploaded chest X-ray images
- apply inference-time preprocessing
- call the hosted model layer
- format hierarchical prediction responses
- return confidence/probability information and disclaimers

The backend should stay independent from frontend concerns. Frontend applications should call the backend API rather than model endpoints directly.

