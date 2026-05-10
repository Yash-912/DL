# Project Phoenix Guidelines

## Overview
Project Phoenix is our internal initiative to migrate all legacy monolithic applications into a microservices architecture.
The target completion date for Phase 1 is Q4 2026.

## Architecture
We use Kubernetes for orchestration. All services must be containerized using Docker.
The database layer is migrating from Oracle to PostgreSQL.

## Deployment
Deployments are handled by ArgoCD. No manual deployments to production are allowed.
Team members must have their PRs approved by at least two senior engineers before merging.
