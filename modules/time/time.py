import commune as c

class Time(c.Module):

    @classmethod
    def timestamp_to_iso(cls, timestamp = c.time(), **kwargs):
        import datetime
        # Convert timestamp to datetime object
        dt = datetime.datetime.fromtimestamp(timestamp, **kwargs)

        # Format datetime object as ISO date string
        iso_date = dt.date().isoformat()

        return iso_date