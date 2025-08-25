import { useEffect, useState } from 'react';

const generateMongoLikeId = () => {
  // Generate 24 character hex string like MongoDB ObjectId
  const timestamp = Math.floor(new Date().getTime() / 1000).toString(16).padStart(8, '0');
  const randomPart = Math.random().toString(16).slice(2, 14).padStart(16, '0');
  return timestamp + randomPart;
};

export const useDeviceId = () => {
  const [deviceId, setDeviceId] = useState<string>('');

  useEffect(() => {
    let id = localStorage.getItem('device_id');
    if (!id) {
      id = generateMongoLikeId(); // Use our custom generator instead of uuidv4
      localStorage.setItem('device_id', id);
    }
    setDeviceId(id);
  }, []);

  return deviceId;
}; 