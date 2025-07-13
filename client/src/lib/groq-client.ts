// This file contains the client-side utilities for GROQ integration
// The actual API calls are handled by the backend service

export const BIAS_TYPES = {
  GENDER: 'gender',
  RACIAL: 'racial', 
  POLITICAL: 'political',
  CULTURAL: 'cultural',
  COMPREHENSIVE: 'comprehensive'
} as const;

export const SENSITIVITY_LEVELS = {
  LOW: 'low',
  STANDARD: 'standard',
  HIGH: 'high'
} as const;

export const RISK_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high'
} as const;

export function getBiasTypeLabel(type: string): string {
  const labels: Record<string, string> = {
    gender: 'Gender Bias',
    racial: 'Racial Bias',
    political: 'Political Bias',
    cultural: 'Cultural Bias',
    comprehensive: 'Comprehensive Analysis'
  };
  return labels[type] || type;
}

export function getSensitivityLabel(level: string): string {
  const labels: Record<string, string> = {
    low: 'Low Sensitivity',
    standard: 'Standard',
    high: 'High Sensitivity'
  };
  return labels[level] || level;
}

export function getRiskLevel(score: number): string {
  if (score <= 3) return RISK_LEVELS.LOW;
  if (score <= 6) return RISK_LEVELS.MEDIUM;
  return RISK_LEVELS.HIGH;
}

export function getRiskColor(level: string): string {
  const colors: Record<string, string> = {
    low: 'text-green-600',
    medium: 'text-yellow-600',
    high: 'text-red-600'
  };
  return colors[level] || 'text-gray-600';
}

export function getRiskBadgeColor(level: string): string {
  const colors: Record<string, string> = {
    low: 'bg-green-100 text-green-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-red-100 text-red-800'
  };
  return colors[level] || 'bg-gray-100 text-gray-800';
}
