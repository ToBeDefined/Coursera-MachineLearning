function [email_contents] = preProcessEmail(email_contents)

email_contents = lower(email_contents);
email_contents = regexprep(email_contents, '<[^<>]+>', ' '); % Strip all HTML
email_contents = regexprep(email_contents, '[0-9]+', 'number'); % Handle Numbers
email_contents = regexprep(email_contents, '(http|https)://[^\s]*', 'httpaddr'); % Handle URLS
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr'); % Handle Email Addresses
email_contents = regexprep(email_contents, '[$]+', 'dollar'); % Handle $ sign

end
