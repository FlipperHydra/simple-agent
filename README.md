This AI-agent allows any and all users to add their own tools, prompts and models. 

The logic behind executing these files is quite simple:

A regex handles XML like output that is different for each tool, for example:
<write_tool>
  <argN>
    content
  </argN>
</write_tool>

