"use client";
import React, { useEffect } from "react";
import classNames from "classnames";
import Link from "next/link";
import classes from "./popup.module.css";

export default function Popup({
  className,
  children,
  overlayUrl,
}: {
  className: string,
  children: React.ReactNode | React.ReactNode[],
  overlayUrl: string,
}) {
  useEffect(() => {
    // TODO this mechanism assumes the original margin is 0. We should improve it,
    // but that would require parsing of the original value including the unit
    const scrollBarWidth = window.innerWidth - document.body.offsetWidth;

    document.body.style.overflow = 'hidden';
    document.body.style.marginRight = scrollBarWidth + 'px';

    return () => {
      document.body.style.marginRight = "";
      document.body.style.overflow = 'unset';
    };
  }, []);

  return (
    <Link className={classes.overlay} href={overlayUrl}>
      <div className={classNames(classes.popup, "my-auto mx-auto z-40 bg-gray-100 rounded-lg border-2 border-zinc-700 dark:border-gray-100 border-solid shadow-md", className)} onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </Link>
  );
}
