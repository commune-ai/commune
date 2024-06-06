'use client';

export default function Card({ className, children }: { className?: string; children: React.ReactNode }) {
   return (
      <div className={`${className} drop-shadow-[2px_2px_5px_rgba(0,0,0,0.16)] bg-white rounded-[7px] font-medium`}>
         {children}
      </div>
   );
}
