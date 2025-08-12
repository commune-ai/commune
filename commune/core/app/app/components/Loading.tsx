import React from 'react'

export function Loading() {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-gray-900 dark:bg-gray-900">
      <div className="text-center">
        {/* IBM Carbon Design System spinner */}
        <svg className="animate-spin h-12 w-12 mx-auto" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="16" cy="16" r="14" stroke="#393939" strokeWidth="4"/>
          <path d="M16 2C8.268 2 2 8.268 2 16" stroke="#0f62fe" strokeWidth="4" strokeLinecap="round"/>
        </svg>
        
        {/* Loading text */}
        <p className="mt-4 text-sm text-gray-400">Loading...</p>
        
        {/* Retry button */}
        <button 
          onClick={() => window.location.reload()}
          className="mt-4 px-4 py-2 text-sm bg-gray-800 hover:bg-gray-700 text-gray-100 rounded transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
          aria-label="Retry loading"
        >
          Retry
        </button>
      </div>
    </div>
  )
}