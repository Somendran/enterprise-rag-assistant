import axios, { AxiosHeaders } from 'axios';
import { API_BASE, API_KEY, AUTH_TOKEN_STORAGE_KEY, DEMO_SESSION_STORAGE_KEY } from '../types';

const client = axios.create({
  baseURL: API_BASE,
});

client.interceptors.request.use((config) => {
  const headers = new AxiosHeaders(config.headers);
  const authToken = window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) || '';
  const demoToken = window.localStorage.getItem(DEMO_SESSION_STORAGE_KEY) || '';

  if (API_KEY) headers.set('X-API-Key', API_KEY);
  if (authToken) headers.set('Authorization', `Bearer ${authToken}`);
  if (demoToken) headers.set('X-Demo-Session-Id', demoToken);

  config.headers = headers;
  return config;
});

client.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    if (axios.isAxiosError(error) && error.response?.status === 401) {
      window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
    }
    return Promise.reject(error);
  },
);

export default client;
